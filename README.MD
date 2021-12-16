# Can Less be More? When Increasing-to-Balancing Label Noise Rates Considered Beneficial

This is the source code for the NeurIPS 2021 paper *Can Less be More? When Increasing-to-Balancing Label Noise Rates Considered Beneficial*.

## Data
The pre-processed data including image features and clean labels is available at [https://www.dropbox.com/s/ztkj5o3xenf2nx5/data.tar.gz?dl=0](https://www.dropbox.com/s/ztkj5o3xenf2nx5/data.tar.gz?dl=0).

## Usage
To reproduce the experiments, simply run

```
python3 CanLessBeMore.py --min_noise 0.2 --max_noise 0.4
```

You can set any noise rates as you want by changing the parameters in `min_noise` and `max_noise`