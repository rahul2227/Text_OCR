##  Scripts
This file contains all the scripts necesssary for working this project

### Installing dependecies

```bash
pip install -r requirements.txt
```

### Data preprocessing
For parsing IAM dataset and creating processed data

```bash
python new_parser.py --data_dir ../data/ --fixed_height 128
```

### Running the LSTM model