##  Scripts
This file contains all the scripts necesssary for working this project

### Installing dependecies

```bash
pip install -r requirements.txt
```

### Data preprocessing
For parsing IAM dataset and creating processed data with maximized width images (ONLY DO THIS WHEN A CLUSTER IS AVAILABLE)

```bash
python new_parser.py --data_dir ../data/ --fixed_height 128
```

For a smaller image size - MEMORY OPTIMIZATION
```bash
python new_parser.py --data_dir ../data/ --fixed_height 128 --resize_image_smaller
```

For a image preprocessing with capped image width

```bash
python new_parser.py --data_dir ../data --fixed_height 128 --resize_image_smaller --smaller_max_width 4096
```

> Change the parameter --smaller_max_width 4096 to 2048 for even smaller images

### Running the LSTM model