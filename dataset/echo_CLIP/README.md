# Process Dataset Using EchoClip

```bash
conda create -n echoclip python=3.9
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

python dataset_processing_label.py --save_studies
```