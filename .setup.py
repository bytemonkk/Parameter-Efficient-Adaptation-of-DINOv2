from setuptools import setup, find_packages

setup(
    name="dinov2_lora_segmentation",
    version="1.0.0",
    author="Manoj Kumar Sunkara",
    description="Parameter-Efficient Adaptation of DINOv2 for Remote Sensing Semantic Segmentation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "pandas",
        "opencv-python",
        "Pillow",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "pyyaml"
    ],
)