
# COMP6721 - Final Project Assignment

## Project Structure
Here is our project folder structure and files with their descriptions.

```
root
│   README.md
│   requirements.txt: contains all of the project dependencies (torch, NumPy, Pillow, scikit-learn, pandas, seaborn)
│
└───src
│	|	config.py: contains all configurations from hyper-parameters to random seed and other global constants.
│	|	model.py: contains our CNN model called MaskCNN.
|	|	train.py: contains the training loop and model saving process. 
|	|	test.ipynb: contains test and evaluation of the model along with their figures.
|	|	mask_dataset.py: contains our torch Dataset called MaskDataset. It automatically loads the dataset and generates training/validation dataset loaders plus on-the-fly data augmentations.
|	|	train_folds.ipynb: contains the k-fold cross evaluation code.
|	|	bias_eval.ipynb: contains the code to measure bias in each sub-category and generate the figures.
│   
└───trained
|	|	MaskCNN.pt: Final trained model for Phase 1.
|	|	MaskCNN_unbiased.pt: final unbiased model for Phase 2. 
|
└───dataset
|	|	sources_links.txt: contains links to each dataset.
|	|	sources_metrics.xlsx: contains numbers of each dataset and per class counts.
|	|	metadata.csv: contains age and gender labels for all images in the dataset. 
|	└───celeba_hq
|		└───no-mask
|	└───cfmdd
|		└───cloth, surgical, n95, ...
|	└───googleimages
|	└───hitl
|	└───mfmdd
|	└───wwmr-db
```


## Training

### How-to
Our training loop happens in `src/train.py`. We recommend using the same `conda` environment that has been defined for **COMP6721**. This means using `python 3.6` and having the basic required packages like `torch`, `numpy`, `scikit-learn`, `pandas`, `seaborn` for heatmap, and `Pillow` for image loading. Also, for faster training, you should use a `cuda` GPU with at least `3GB` VRAM and related torch packages installed. Should no GPU is not available, the CPU will be picked for training automatically.

#### Example of How to Run `train.py`:
```
python.exe c:/dev/comp6721_mask_recon/src/train.py
```

### Dataset
Since we have manually cropped our images to have a better data consistency over multiple sources, all the modified images are included as part of our deliverables and would be picked up by the data loader automatically for the training process.


## Testing and Evaluation

We have a notebook in `src` folder called `test.ipynb`. It contains all our testing and evaluation codes along with some figures such as a confusion matrix.  The trained model `trained/MaskCNN_0.pt` would be loaded for evaluation. Running this notebook should give the expected result and is pretty much self-explanatory.

## Authors:
- Mohammadamin Aliari
- Richard Grand'Maison
