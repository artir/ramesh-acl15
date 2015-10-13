package Main;

import BOW.DataReader;
import Sampler.Sampler;

public class Main {
	public static void main(String[] args) {
		Storage.Init();
		DataReader.ReadStopwords("./Words/Stopwords.txt");
		DataReader.ReadAmazonFile("./Amazon/");
		DataReader.ReadSentiWords("./Words/");
		Storage.UpdateBOW();
		Sampler.Go();
	}
}