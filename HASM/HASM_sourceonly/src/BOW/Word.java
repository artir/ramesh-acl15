package BOW;

import Main.Storage;

public class Word {
	public int p; // Word subjectivity.
	// p = 0 when word is neutral
	// p = 1 when word is subjective (either can have positive or negative sentiment)
	
	public int w; // Word index, ranging from 0 <= w <= Storage.V - 1
	// Look for Storage.InvDictionary.get(w) to get Stringified word 
	
	public String toString() {
		return Storage.InvDictionary.get(w);
	}
}
