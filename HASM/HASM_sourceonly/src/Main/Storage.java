package Main;

import java.util.*;

import BOW.Document;
import Hierarchy.Node;

public class Storage {
	////////// BOW //////////
	
	public static List<Document> Documents;
	public static Map<String, Integer> Dictionary;
	public static Map<Integer, String> InvDictionary;
	
	////////// PARAMETERS //////////////
	
	public static int V;
	public static int N;
	public static int MAX_ITER;
	public static List<List<String>> SentiWords;
	
	////////// HYPERPARAMETERS //////////
	
	public static double alpha;
	public static double[] beta_init;
	public static double[][] beta;
	public static double beta_sum;
	public static double gamma;
	public static double eta;
	
	////////// HIERARCHY ////////
	
	public static int NodeCounter = 0;
	public static Node Root;
	public static int InitCPN;
	public static int InitMaxDepth;
	public static Map<Integer, Node> Nodes;
	
	public static void UpdateBOW() {
		N = Documents.size();
		V = Dictionary.size();
		beta = new double[3][V];
		Root = new Node();		
		System.out.println("N = " + N);
		System.out.println("V = " + V);

		// SET BETA
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < V; j++) {
				beta[i][j] = beta_init[1];
			}
		}
		for (int i = 0; i < 3; i++) {
			for (String word : SentiWords.get(i)) {
				for (int k = 0; k < 3; k++) {
					beta[k][Dictionary.get(word)] = beta_init[0];
				}
			}
		}
		for (int i = 0; i < 3; i++) {
			for (String word : SentiWords.get(i)) {
				beta[i][Dictionary.get(word)] = beta_init[2];
			}
		}
		beta_sum = 0;
		for (int v = 0; v < V; v++) {
			beta_sum += beta[0][v];
		}
	}
	
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
}
