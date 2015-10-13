package BOW;
import java.util.ArrayList;
import java.util.List;

public class Document {
	public List<Sentence> Sentences; // List of sentences assigned to this document
	
	public int pi[]; // Counter variable for sentiments
	// pi[k] = n^{s, (k)}_{this}
	// pi[0] is not used
	// e.g. pi[1] is the number of positive sentences in this document
	
	public Document() {
		Sentences = new ArrayList<Sentence>();
		pi = new int[3];
	}
}