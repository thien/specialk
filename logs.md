# TODO

- Check for any teacher forcing implemention differences between the models.
- Note to self: Don't merge the optimisation methods. They're inherently different (see Attn is all you need paper.)

---

## Methods

- [ ] Setup early stopping to 2/3 of best outcome (will this affect the bayesian optimiser?)
- [x] Save epoch in models (for checkpointing).
- [ ] Need mechanism to send data to local machine once done.
    - [x] Write `rsync` script to handle transfer to local machine.
    - [x] Add telegram notifier to tell me when it's time to turn off
          the azure instance.
        - [x] Move telegram component to outside models s.t it can be used with `bash`.
    - [ ] Test it on the azure instance.
- [ ] Need to fix seq2seq model
    - [ ] update save method to match transformer.
    - [ ] fix model training (it's quite broken.)
- [ ] Need to research how to deal with different batch sizes and sequence lengths.
    - [ ] Need mechanism to deal with detecting memory requirements based on batch and sequence length.
    - [ ] Experiment with only lowercase sequences to optimise memory requirements.
    - [x] Need to consider how to process newspaper articles in metrics.py. Possibly replacing \n tags with a custom token representing a new paragraph.
- [ ] Train en-fr model and vice versa.
    - [x] Create dataset.
    - [ ] Disect prabhumboye en-fr model and compare differences.
    - [ ] Actually start training the model. (You should build the telegram notifier while you're at it.)
- [x] Read Training Tips for the Transformer Model.
- [ ] Fix metrics mechanism (currently experimenting with [`nlg-eval`](https://github.com/Maluuba/nlg-eval) but it's broken somewhere.)
- [ ] Test that you can train only the decoder.
- [ ] Setup back-translation dataset for our transformer model.

# History

## 7/09

Note: Since we're just checking whether it's even possible to do style transfer with transformers, don't worry about the optimisation part yet. That being said, it's built and ready for use when needed. I should worry about getting models out first.

- Updated metrics.py mechanism
- Initial commit of built bayesian optimiser wrapper script `optimise.py`.
- TODO: wrap it for seq2seq support
- TODO: need to test training a new decoder only (with a encoder already existing.)
- TODO: write batch operations for bayesian optimisation for small, medium and large EN-FR models.
- TODO: create FR-EN models.

Note:
- train en-fr with raw
- test: en(styleset)-fr
- train fr-en with raw
- train fr-decoder with dataset

- TODO: need method to save best_model_dir in bayeopt
- make sure we have the style datasets.
- need to fix metrics (since the installer is broken for some reason)

## 7/08

- Initial commit of metrics.py build.
- Looking into relevant hyperparameter optimisation approaches. (https://botorch.org/tutorials/optimize_stochastic)

## 7/04

- Recurrent trainer needs major refactoring
- TODO: Reverse engineer the beam mechanism for the seq2seq translator.
- TODO: Telegram support
- TODO: Bayesian Hyperoptimisation

## 7/02

- Implemented Transformer Trainer
- Implemented Seq2Seq Trainer
- Built batch functions for relevant dataset parsers
- Built dataset downloaders.
- Finished integrating the trainer classes into respective functions.
- Built model trainer functions.
- Initial tests on training data on en-de 30k shows that the seq2seq model greatly surpasses the transformer model. We'll have to build translator components to verify that this is the case. (Quite strange!)

## 6/20

- Added `preprocess.py` to deal with turning mose style data into something that can be interpreted 
  by the models.
- Added support for byte-pair encoding thanks to some open source code on GitHub (cites are there).

## 6/19

- Added `splitter.py` to deal with breaking datasets.
- Added `splitter.py` command to run in `initiate.sh`.

## 6/18

- built data cleaning mechanism for global voices, hansards, and europarl dataset. 
- created bash script to download the global voices dataset, and deal with cleaning it. Also the case for hansards and europarl.
- next steps would be to write some code to tokenise the datasets and create a feeder to a model.
- using moses tokeniser perl code to tokenise [french](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.fr) and [english](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en).
- added moses tokeniser source code.
- TODO: need code to deal with splitting the dataset into train valid test.

## 6/17

- Finished rerunning the models to verify source code integrity for the style transfer through back translation method. It does seem to work but as mentioned earlier, there isn't a copy of the original machine translation dataset they used. They only provided a pre-trained model. This might be because it took them forever to train.
- We'll need to start with making our own machine-translation dataset. Since this alone is (although "solved") difficult we'll have to determine the most appropiate method of training the models.
- We'll need to have models to look like the following:

        seq2seq model
        - sentence level
        - paragraph level
        transformer model
        - sentence level
        - paragraph level
        - document level

- Fortunately, we have access to some datasets that might be able to help us with sentence level. Paragraph level gets a bit trickier. 
- Currently using:
    - europarl
    - [hansards](https://www.isi.edu/natural-language/download/hansard/)
    - global voices (need to filter)


## 6/14

- Retrained the models and the error remains. Since the majority of this codebase is taken from `OpenMNT-py` on GitHub I'm looking into porting the relevant code from there and using that instead.
- However, some data from the original paper is missing (e.g. the valuation dataset for the neural classifier) so for now I'm spliting `classtrain.txt` since there is around 80K sequences. Currently using a 4:1 split. Currently writing writing split code.
- Ported `style_transfer/classifier` and translated any deprecated code from PyTorch.
- Trained the classifier.

## 6/13

- More refactoring in the models as `train_decoder.py` wasn't calculating softmax values in the right dimension (since the default values are deprecated from an earlier version of PyTorch.)
- Had to recompute some of the model outputs since they were corrupted (computer randomly shut off).
- Presently there's an issue with the encoder model reporting an accuracy of 0%, looking into some of the outputs it looks like it's reporting issues with predicting the right output which may correlate with the erroneous softmax outputs. We'll have to see if it changes after recomputing the model outputs.
- Since the models output the responses to a plaintext file it shouldn't be difficult to port the models into the transformer.

## 6/12

- Waiting for models to finish translating the datasets; moved to GPU now.
- Looking into large length machine translation datasets for eventual use with the transformers for back translation:
    - [Canadian Hansards](https://www.isi.edu/natural-language/download/hansard/)
    - [EUROPARL](http://www.statmt.org/europarl/)
    - [Global Voices](https://casmacat.eu/corpus/global-voices.html) is the most intriguing one since it's a parallel corpus of news articles. This may help to retain more nuanced information such at the lexical relationships between nouns within the same paragraph for instance.
- _Note to self:_ Once the models are trained, you'll want to explore the OMT models in more detail since they only have the trained weights and the source code. 

## 6/11

- Downloaded 40K Guardian articles.
- Started refactoring Back Translation code since it was built for an earlier version of PyTorch. Currently working on porting the OpenMNT parts of the code (for now, translation). 
- Retraining models to get french political data. (They're not provided in the datasets).

## 6/10

- Added readability scores, word mover's distance, and mechanisms to load SpaCy and GloVe datasets.
- Added syntactic analysis.
- Experimented on optimal methods to load the word2vec dict into memory.
- Explored [Guardian](https://open-platform.theguardian.com/documentation/) and [New York Times](https://developer.nytimes.com/) APIs for scraping use.
- Built syllable counting mechanisms.
- TheGuardian has a nice API that allows us to scrape articles by search format; we can integrate this with our scraper and we should be able to scrape a large corpus from the Guardian.
- Presently these are the number of articles we've managed to scrape to date:
	
	thesun           2029 (+0)                                                                         
	mirror           2480 (+0)                                                                        
	theguardian      977 (+0)                                                 
	bbc              114431 (+0)                                              
	reuters          530 (+0)                                                 
	thetimes         3517 (+0)                                                        
	dailymail        9114 (+0)                                                        
	metro            3305 (+0)                                                       
	telegraph        1290 (+0)   
