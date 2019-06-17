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
    - [Global Voices](casmacat.eu/corpus/global-voices.html) is the most intriguing one since it's a parallel corpus of news articles. This may help to retain more nuanced information such at the lexical relationships between nouns within the same paragraph for instance.
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
