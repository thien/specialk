## 7/10

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
