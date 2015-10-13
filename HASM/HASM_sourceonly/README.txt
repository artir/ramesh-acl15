========================================================
A Hierarchical Aspect-Sentiment Model for Online Reviews
========================================================

Suin Kim
suin.kim[at]kaist.ac.kr

(C) Copyright 2014, Suin Kim
Last updated Aug 26 2014


This source code is a personal, reimplemented version based on the publicly
available paper, "A Hierarchical Aspect-Sentiment Model for Online Reviews", 
published in AAAI-13.

Included Amazon corpus is crawled by Yohan Jo (yohan.jo[at]kaist.ac.kr). This
data is originally used by the paper "Aspect and Sentiment Unification Model
for Online Review Analysis" (WSDM'11).

DISCLAIMER: This program is distributed in the hope that 
it will be useful, but WITHOUT ANY WARRANTY expressed or implied, 
including the implied warranties of MERCHANTABILITY or FITNESS FOR 
A PARTICULAR PURPOSE.

----------------------------------------------------------

TABLE OF CONTENTS

A. COMPILING

   1. RUNNING

B. DATA STRUCTURE
    
   1. Document
   
   2. Sentence
   
   3. Word

C. PARAMETERS

-----------------------------------------------------------

A. COMPILING

Please include all source files, as well as args4j library (whch I included),
and compile with standard Java compiling options. On development, I used 
Eclipse Luna (4.4.0) with JDK 1.8.0_20.

A.1. RUNNING

Be sure to include Words/ directories on the same directory so that HASM can
look up the list of stopwords and sentiment seed words.

By default, stopwords file is set to be located at

./Words/Stopwords.txt

and sentiment seed words are set to be located at 

./Words/SentiWords-0.txt
./Words/SentiWords-1.txt
./Words/SentiWords-2.txt

Finally, I implemented document parser for included set of Amazon.com reviews,
which is located at BOW/DataReader.java/ReadAmazonReviews(). Note that you need
to reimplement the function to fit your data.



B. DATA STRUCTURE

As of standard latent Dirichlet allocation framework, each instance is
represented as a Document. Each document contains sentences, which contains
list of words. Document, Sentence, Word are implemented as of single class:

Document: BOW/Document.java
Sentence: BOW/Sentence.java
Word: BOW/Word.java

Current version prints topic hierarchy over console per 10 iterations. To get
detailed information, such as polarity of words or sentiment of sentences, you
can access Document, Sentence and Word class.

B.1. Document

public class Document {
	public List<Sentence> Sentences; // List of sentences assigned to this document
	
	public int pi[]; // Counter variable for sentiments
	// pi[k] = n^{s, (k)}_{this}
	// pi[0] is not used
	// e.g. pi[1] is the number of positive sentences in this document
}

B.2. Sentence

public class Sentence {
	public List<Word> Words; // List of Words
	
	public int s; // Sentiment of this sentence
	// s will NOT have 0 value
	// s = 1: positive
	// s = 2: negative
	
	public int theta[]; // Counter variable for polarity
	// theta[k] = n^{p, (k)}_{d, this}
	// e.g. theta[0] is the number of neutral words in this sentence
	
	public Node node; // Aspect-sentiment node this sentence is assigned
}

B.3. Word

public class Word {
	public int p; // Word subjectivity.
	// p = 0 when word is neutral
	// p = 1 when word is subjective (either can have positive or negative sentiment)
	
	public int w; // Word index, ranging from 0 <= w <= Storage.V - 1
	// Look for Storage.InvDictionary.get(w) to get Stringified word 
}


C. PARAMETERS

Most of the parameters, including hyperparameters and Gibbs sampling options,
are located at init() function in Storage.java. Be sure to change the 
parameters and options to get the best results.

public static void Init() {
	// BOW
	Documents = new ArrayList<Document>();
	Dictionary = new HashMap<String, Integer>();
	InvDictionary = new HashMap<Integer, String>();
	
	// PARAMETERS
	MAX_ITER = 1000; // Number of iteration for Gibbs Sampling process
	
	// HYPERPARAMETERS
	alpha = 25.0; // Alpha, Dirichlet hyperparameter for theta
	gamma = 0.01; // Gamma, rCRP prior
	beta_init = new double[3];
	beta_init[0] = 1e-9; // Beta for sentiment-topic nonaligned words
	beta_init[1] = 0.001; // Beta for neutral words
	beta_init[2] = 2.000; // Beta for sentiment-topic aligned words
	eta = 1.0; // Eta, Dirichlet hyperparameter for pi
	
	// HIERARCHY
	Nodes = new HashMap<Integer, Node>();
	InitCPN = 3; // Number of children per node for initial random sampling
	InitMaxDepth = 2; // Max depth for initial random sampling
}