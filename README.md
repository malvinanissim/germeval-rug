# germeval-rug

Repository of the [RuG Team](https://sites.google.com/view/sms-rug) @ Germeval 2018 for the [Shared Task on the Identification of Offensive Language](https://projects.fzai.h-da.de/iggsa/).

Structure of the repository:

- the Data folder contains the training data from the organizers as well as the additional data we used for training, drawn from the German section of the [Political Speech Project](https://rania.shinyapps.io/PoliticalSpeechProject/), called "espresso data" here. Use either all of the German espresso data or only the ones labelled as offensive. The extra data are available as pickled Python objects. With respect to the shared task data by the organizers, the folder contains the official training, testing data and a small sample dataset not used in the actual task ("germeval2018.training.txt", "germeval2018.test.txt" and "germeval2018.sample.txt"). For validation of models, the official training data has been given a 80%/20%-train/dev-split, respectively called "germeval.ensemble.train.txt" and "germeval.ensemble.test.txt". To run our models, you will also need to download our "hate" and "hate-oriented" embeddings from [here](https://drive.google.com/drive/u/0/folders/12muUeRfs2FSy3Wfd2OHbViOaLIfPENWH?ogsrc=32). The 300D hate embeddings are used in our CNN model and the 52D hate-oriented ones in our SVM (please refer to our paper for details).

- the Models folder contains the scripts to train the models
   * Baseline: This contains our baselines models (majority class and linear SVM with Tfidf-weighted word unigrams), for comparison only.
   * SVM: Our SVM models. "SVM_final_runs.py" trains on a training dataset and outputs predictions for a given test dataset. To run it please change the paths for the input data accordingly. The offensive terms in "lexicon.txt" were copied from [this website](http://www.hyperhero.com/de/insults.htm).
   * CNN: Our CNN model. The input data files are to be specified in the "load_data_and_labels()" function in the "data_helpers.py" script, and the embeddings path is in the w2v_xy.py files. Run "CNN_get_dev.py" to train on the training data (specified in data_helpers) and evaluate on the test data.
   * Ensemble: Our Ensemble model. See below for instruction on how to run them.   

- the Resources folder contains scripts we used to concatenate different sets of word embeddings. These were not included in our final submissions to the shared task  

- the Results folder contains the results of some of our experimental results obtained on a randomly chosen dev set. They show how different sets of word embeddings, including our hate-oriented ones, perform in combination with the linear SVM model. 
   
- the Submissions folder contains the 4 runs we submitted to the shared task (and the Perl evaluation script provided by the organizers). Those marked "coarse" deal with the binary classification task (OFFENSE vs. NONE), the single one marked with "fine" deals with the 4-class task (INSULT, PROFANITY, ABUSE, NONE).

To run the Ensemble model you should follow these steps:
- Run the CNN and SVM cross-prediction scripts first. They will output predictions of the CNN and SVM over the trainig data. The output is stored in pickle files (names starting with  "NEW-train")
- Run the CNN and SVM single prediction scripts. They will output the predictions of the CNN and SVM over the test data. The output is stored in pickle files (names starting with "TEST")
- Run the ensemble.py script which reads in the CNN and SVM predictions, then trains and tests the logistic regression meta-classifier.


All scripts are based on Python 3.5 / 3.6. CNN uses Keras 2.2.0 and can be run with Tensorflow 1.8.0 as backend.



# Citation
If you find any of these pre-trained models useful, please cite the following papers: 
- [RuG at GermEval: Detecting Offensive Speech in German Social Media](https://github.com/malvinanissim/germeval-rug/blob/master/rug-germeval-paper.pdf)
```
@InProceedings{RUG@Evalita2018,
  author    = {Xiaoyu Bai and Flavio Merenda and Claudia Zaghi and Tommaso Caselli and Malvina Nissim},
  title     = {{RuG at GermEval: Detecting Offensive Speech in German Social Media}},
  booktitle = {Proceedings of the GermEval 2018 Workshop},
  year      = {2018},
  Editor = {Josef Ruppenhofer and Melanie Siegel and Michael Wiegand}
  address   = {Wien, Austria}
  Publisher = {{ÖAW Austrian Academy of Sciences}}
}
``` 
