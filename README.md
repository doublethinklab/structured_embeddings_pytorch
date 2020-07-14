# Structured, Dynamic Embeddings

## Docker Notes

For training, use GPU, so need nvidia docker.
To install locally see [instructions here](https://github.com/NVIDIA/nvidia-docker).

## Notes on original repository

Lives [here](https://github.com/mariru/structured_embeddings).

### Data Preprocessing

#### 1. [Count words](https://github.com/mariru/structured_embeddings/blob/master/dat/step_1_count_words.py)

- Tokenization
- Create word dict
- Convert token stream to integer stream and save for later use
- Generate counts for all tokens
- Reorder dictionary by count
- Take the top-V frequent tokens, dropping the rest

#### 2. [Split data](https://github.com/mariru/structured_embeddings/blob/master/dat/step_2_split_data.py)

- Convert counts to freqs
- Subsample frequent words (follows [Mikolov](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) p. 4)

$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}, \quad t = 10^{-5}
$$

```python
dat = np.load(fname)
prob = np.random.uniform(0,1,dat.shape)
p = 1 - np.sqrt((10.0**(-5))/cnt[dat])
dat = dat[prob > p]
```

- Split the resulting stream of word indices into:
  * train: 80%
  * validation: 10%
  * testing: 10%
  
```python
split = int(0.1*len(dat))
i = np.random.randint(len(dat))

dat = np.roll(dat, i)
test_dat = dat[:split]
dat = dat[split:]

dat = np.roll(dat, i)
valid_dat = dat[:split]
dat = dat[split:]
```

#### 3. [Create data stats](https://github.com/mariru/structured_embeddings/blob/master/dat/step_3_create_data_stats.py)

The `dat_stats` dict has: 
- `name`: name of the dataset.
- `T_bins`: list of states.
- `split` (train, valid, test): np vector of length `num_groups`, and each entry
  indicates the size of that dataset (i.e. how many tokens, as I understand it).

#### 4. [Negative sampling](https://github.com/mariru/structured_embeddings/blob/master/dat/step_4_negative_samples.py)


