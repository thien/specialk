wget http://tts.speech.cs.cmu.edu/style_models/english_french.tar
wget http://tts.speech.cs.cmu.edu/style_models/french_english.tar
cd ..
mkdir -p classifier
cd classifier
wget http://tts.speech.cs.cmu.edu/style_models/gender_classifier.tar
wget http://tts.speech.cs.cmu.edu/style_models/political_classifier.tar
wget http://tts.speech.cs.cmu.edu/style_models/sentiment_classifier.tar
cd ..
mkdir -p style_generators
cd style_generators
wget http://tts.speech.cs.cmu.edu/style_models/female_generator.tar
wget http://tts.speech.cs.cmu.edu/style_models/male_generator.tar
wget http://tts.speech.cs.cmu.edu/style_models/democratic_generator.tar
wget http://tts.speech.cs.cmu.edu/style_models/republican_generator.tar
wget http://tts.speech.cs.cmu.edu/style_models/positive_generator.tar
wget http://tts.speech.cs.cmu.edu/style_models/negative_generator.tar
cd ..
cd ..
wget http://tts.speech.cs.cmu.edu/style_models/political_data.tar
tar -xvf political_data.tar
