# germeval-rug

Repository of the [RuG Team](https://sites.google.com/view/sms-rug) @ Germeval 2018 for the [Shared Task on the Identification of Offensive Language](https://projects.fzai.h-da.de/iggsa/).

Structure of the repository:

- the Data folder contains the training data from the organizers. Files with FB refers to the Facebook data, files with TW refers to the Twitter dataset. Test data are currently available at [this link](https://docs.google.com/forms/d/e/1FAIpQLSeJXpDygWFeiFxldm2J0OuNyQBp5yDxlAgc7200fnycaEyU2A/viewform). The files marked with "ensamble" are used to obaine the cross-fold prediction from the CNN and the SVM models to train the meta-classifier (Logistic Regressor) for the Ensamble models. The embeddings folder contains only the .bin file of the Italian Hate Embeddings obtaine using the skip-gram model from word2vec. To run the models you need to download [this file](https://drive.google.com/drive/folders/133EPm4mO9dN6A0Cw6A6Sx1ABa-25BI8e?usp=sharing).  

- the Models folder contains the script to train the models and apply them on the test data. For the SVM and CNN models, we have 4 different models, sharing the same features, but differntiating only for the train and test data they are using. To run the model, please change the paths for file inputs and word embeddings. Foer the CNN models, the input files are specified in the data_helpers scripts, and the embeddings path is in the w2v.py files.  

- the Results folder contains the output of the trained models (CNN, SVM, ensamble) over the trainig data, following the three subtasks:
   
   1) Train: Facebook - Test: Facebook (files with FB-FB or FB only in their names)
   2) Train: Twitter - Test: Twitter (files with TW-TW or TW only in their names)
   3) Train Facebook - Test: Twitter (files with FB-TW in thei names)
   4) Train: Twitter - Test: Facebook (files with TW-FB
   
 - The Submissions-Haspeede folde contains the submissions to all subtasks of the Haspeede Task. Files ending with *run1* correspond to the SVM predictions, files ending with *run3* correspond to the Ensemble model predictions.

Replicating the experiments for the CNN and SVM models is straighforward: modify the path for of the train and test data, of the embeddings (do not forget to [download the embedding file](https://drive.google.com/drive/folders/133EPm4mO9dN6A0Cw6A6Sx1ABa-25BI8e?usp=sharing)), and run the scripts. 
For the Ensamble models you shoudl follow this stesps:
- Run the CNN and SVM cross-prediction scripts first. They will output predictions of the CNN and SVM over the trainig data. The output is stored in pickle files (names starting with  "NEW-Train")
- Run the CNN and SVM single prediction scripts. They will output the predictions of the CNN and SVM over the test data. The output is stored in pickle files (names starting with "TEST")
- Run the ensamble.py script

The ensemble model benefits from 2 extra features: lenght of the text and presence of offensive/hate words. The list of offensive/hate words is stored in the file *lexicon_deMauro.txt*. The file contains stemmed entries obtained from these two resources: *[Le parole per ferire](https://www.internazionale.it/opinione/tullio-de-mauro/2016/09/27/razzismo-parole-ferire)* and [Wikitionary](https://it.wiktionary.org/wiki/Categoria:Parole_volgari-IT).

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
