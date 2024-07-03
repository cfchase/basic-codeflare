import os
import tempfile

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import lightning.pytorch as pl

import ray.train.lightning
from ray.train.torch import TorchTrainer

# Based on https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html

"""
# For S3 persistent storage replace the following environment variables with your AWS credentials then uncomment the S3 run_config
# See here for information on how to set up an S3 bucket https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html

os.environ["AWS_ACCESS_KEY_ID"] = "XXXXXXXX"
os.environ["AWS_SECRET_ACCESS_KEY"] = "XXXXXXXX"
os.environ["AWS_DEFAULT_REGION"] = "XXXXXXXX"
"""

"""
# For Minio persistent storage uncomment the following code and fill in the name, password and API URL then uncomment the minio run_config.
# See here for information on how to set up a minio bucket https://ai-on-openshift.io/tools-and-applications/minio/minio/

def get_minio_run_config():
   import s3fs
   import pyarrow.fs

   s3_fs = s3fs.S3FileSystem(
       key = os.getenv('MINIO_ACCESS_KEY', "XXXXX"),
       secret = os.getenv('MINIO_SECRET_ACCESS_KEY', "XXXXX"),
       endpoint_url = os.getenv('MINIO_URL', "XXXXX")
   )

   custom_fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))

   run_config = ray.train.RunConfig(storage_path='training', storage_filesystem=custom_fs)
   return run_config
"""


# Model, Loss, Optimizer
class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)
        self.log("loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


def train_func():
    # Data
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Training
    model = ImageClassifier()
    # [1] Configure PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=10,
        devices="auto",
        accelerator="auto",
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
        # [1a] Optionally, disable the default checkpointing behavior
        # in favor of the `RayTrainReportCallback` above.
        enable_checkpointing=False,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader)


# [2] Configure scaling and resource requirements. Set the number of workers to the total number of GPUs on your Ray Cluster.
scaling_config = ray.train.ScalingConfig(num_workers=3, use_gpu=True)

# [3] Launch distributed training job.
trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    # run_config = ray.train.RunConfig(storage_path="s3://BUCKET_NAME/SUB_PATH/", name="unique_run_name") # Uncomment and update the S3 URI for S3 persistent storage.
    # run_config=get_minio_run_config(), # Uncomment for minio persistent storage.
)
result: ray.train.Result = trainer.fit()

# [4] Load the trained model.
with result.checkpoint.as_directory() as checkpoint_dir:
    model = ImageClassifier.load_from_checkpoint(
        os.path.join(
            checkpoint_dir,
            ray.train.lightning.RayTrainReportCallback.CHECKPOINT_NAME,
        ),
    )
