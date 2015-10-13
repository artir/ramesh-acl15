package BOW;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import Main.Storage;
import Util.Stemmer;

public class DataReader {
	private static Stemmer stemmer = new Stemmer();
	private static List<String> Stopwords = new ArrayList<String>();
	
	public static void ReadAmazonFile(String dirpath) {
		// Edit as you want.
		// Currently set for Amazon file.
		System.out.println("Reading Amazon reviews...");
		try {
			File dir = new File(dirpath);
			File[] files = dir.listFiles();
			
			for (File file: files) {
				BufferedReader br = new BufferedReader(new FileReader(file.getAbsolutePath()));
				
				Document newDoc = new Document();
				
				String str = "";
				for (int i = 0; i < 10; i++) br.readLine();
				while ((str = br.readLine()) != null) {
					String[] outersplit = str.split("\\.|\\?|\\!");					
					for (String sentence : outersplit) {
						sentence = sentence.trim().toLowerCase().replaceAll("[^a-z]", " ");
						sentence = sentence.replaceAll("\\s+", " ");						
						Sentence newSentence = new Sentence();
						String[] innersplit = sentence.split(" ");
						for (String token : innersplit) {
							token = Stem(token);
							if (token.length() < 2) continue;
							if (Stopwords.contains(token)) continue;
							Word newWord = new Word();
							newWord.w = GetWordIndex(token);
							newSentence.Words.add(newWord);
						}
						if (newSentence.Words.size() >= 2) {
							newDoc.Sentences.add(newSentence);
						}
					}
				}
				
				Storage.Documents.add(newDoc);
				
				br.close();
			}
			System.out.println("Done.");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static String Stem(String word) {
		stemmer.add(word.toCharArray(), word.length());
		stemmer.stem();
		return stemmer.toString();
	}
	
	public static void ReadStopwords(String filepath) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(filepath));
			
			String str = "";
			while((str = br.readLine()) != null) {
				Stopwords.add(Stem(str));
			}
			
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void ReadSentiWords(String filedir) {
		Storage.SentiWords = new ArrayList<List<String>>();
		for (int i = 0; i < 3; i++) {
			Storage.SentiWords.add(new ArrayList<String>());
			String filepath = filedir + "SentiWords-" + i + ".txt";
			try {
				BufferedReader br = new BufferedReader(new FileReader(filepath));
				String str;
				while ((str = br.readLine()) != null) {
					str = str.trim();
					if (str.length() == 0) continue;
					Storage.SentiWords.get(i).add(str);
					GetWordIndex(str);
				}
				br.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	static int GetWordIndex(String word) {
		if (Storage.Dictionary.containsKey(word)) {
			return Storage.Dictionary.get(word);
		}
		int index = Storage.Dictionary.size();
		Storage.Dictionary.put(word, index);
		Storage.InvDictionary.put(index, word);
		return index;
	}
}
