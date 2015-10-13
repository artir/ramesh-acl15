package BOW;
import java.util.ArrayList;
import java.util.List;

import Hierarchy.Node;

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
	
	public Sentence() {
		Words = new ArrayList<Word>();
		theta = new int[2];
		node = null;
	}
}
