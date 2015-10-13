package Sampler;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import BOW.*;
import Hierarchy.*;
import Main.*;
import Util.Gamma;
import Util.Printer;
import Util.Probability;

public class Sampler {
	
	static Random RAND = new Random(0);
	static Node Root = Storage.Root;
	static Node EmptyNode;
	
	public static void Go() {
		makeInitTree(Storage.Root, 0);
		randomAssign();
		EmptyNode = new Node();
		System.out.println(Storage.NodeCounter);
		GibbsSample();
	}
	
	private static void GibbsSample() {
		for (int ITER = 0; ITER < Storage.MAX_ITER; ITER++) {
			if (ITER % 10 == 0) System.out.println("Iteration " + ITER);
			else System.out.print(".");
			for (Document d : Storage.Documents) {
				for (Sentence s : d.Sentences) {
					SampleAspect(s, d);
					SampleSentiment(s, d);
					for (Word w : s.Words) {
						SampleSubjectivity(w, s, d);
					}
				}
			}
			if (ITER % 10 == 0) {
				Printer.PrintToConsole(Storage.Root, 0);
			}						
		}
	}
	
	private static void SampleAspect(Sentence s, Document d) {
		assignS(-1, s, d, s.node);
		Node prevNode = s.node;
		s.node = rCRP(s, Root);
		assignS(1, s, d, s.node);
		if (prevNode.M == 0) prevNode.RemoveMe();		
	}
	
	private static Node rCRP(Sentence s, Node n) {
		double[] logP = new double[2 + n.Children.size()];
		// 0. SELECT THE CURRENT NODE
		logP[0] = Math.log(n.m) + getSentenceLogProb(s, n);
		// 1. SELECT A CHILD C_k
		for (int i = 0; i < n.Children.size(); i++) {
			Node child = n.Children.get(i);
			logP[i+1] = Math.log(child.M) + getSentenceLogProb(s, child);
		}
		// 2. CREATE A NEW CHILD
		logP[n.Children.size() + 1] = Math.log(Storage.gamma) + getSentenceLogProb(s, EmptyNode);
		
		int index = Probability.SampleLogMultinomial(logP);
		if (index == 0) { return n; }
		else if (index == n.Children.size() + 1) {
			logP[0] = Math.log(n.m) + getSentenceLogProb(s, n);
			Probability.SampleLogMultinomial(logP);
			return n.AddChild();
		} else { return rCRP(s, n.Children.get(index-1)); }
	}
	
	private static void SampleSentiment(Sentence s, Document d) {
		double[] p = new double[3];
		p[0] = Double.NEGATIVE_INFINITY;
		assignS(-1, s, d, s.node);
		
		for (int senti = 1; senti <= 2; senti++) {
			p[senti] = Math.log(Storage.eta + d.pi[senti]);
			s.s = senti;
			p[senti] += getSentenceLogProb(s, s.node);
		}
		s.s = Probability.SampleLogMultinomial(p);
		
		assignS(1, s, d, s.node);
	}
	
	private static void SampleSubjectivity(Word w, Sentence s, Document d) {
		assignW(-1, w, s, s.node);
		
		double[] p = new double[2];
		for (int subj = 0; subj <= 1; subj++) {
			p[subj] = Storage.alpha + s.theta[subj];
			Topic topic = s.node.Topics[subj * s.s];
			p[subj] *= (double)(topic.count[w.w] + Storage.beta[subj*s.s][w.w]);
			p[subj] /= (double)(Storage.beta_sum + topic.n_k);
		}
		w.p = Probability.SampleMultinomial(p);
		
		assignW(1, w, s, s.node);
	}
	
	private static double getSentenceLogProb(Sentence s, Node n) {
		double prob = 0;
		
		List<Map<Integer, Integer>> polarWords = new ArrayList<Map<Integer, Integer>>();
		for (int i = 0; i < 2; i++) {
			polarWords.add(new HashMap<Integer, Integer>());
		}
		for (Word w : s.Words) {
			Map<Integer, Integer> map = polarWords.get(w.p);
			if (!map.containsKey(w.w)) map.put(w.w, 0);
			map.put(w.w, map.get(w.w)+1);
		}
		
		for (int subj = 0; subj <= 1; subj++) {
			int senti = s.s * subj;
			Topic k = n.Topics[senti];
			prob += Gamma.logGamma(k.n_k + Storage.beta_sum);
			prob -= Gamma.logGamma(k.n_k + Storage.beta_sum + s.theta[subj]);
			for (int v : polarWords.get(subj).keySet()) {
				prob += Gamma.logGamma(k.count[v] + Storage.beta[senti][v] + polarWords.get(subj).get(v));
				prob -= Gamma.logGamma(k.count[v] + Storage.beta[senti][v]);
			}
		}
		polarWords.clear();
		polarWords = null;
		return prob;
	}
	
	private static void makeInitTree(Node current, int depth) {
		if (depth == Storage.InitMaxDepth) return;
		while (current.Children.size() < Storage.InitCPN) {
			makeInitTree(current.AddChild(), depth + 1);
		}
	}
	
	private static void randomAssign() {
		for (Document d : Storage.Documents) {
			// Nothing to do here
			for (Sentence s : d.Sentences) {
				s.s = RAND.nextInt(2) + 1;
				d.pi[s.s]++;
				s.node = Storage.Nodes.get(RAND.nextInt(Storage.NodeCounter));
				for (Word w : s.Words) {
					w.p = RAND.nextInt(2);
					s.theta[w.p]++;
				}
				assignS(1, s, d, s.node);
			}
		}
	}
	
	private static void assignS(int flag, Sentence s, Document d, Node n) {
		d.pi[s.s] += flag;
		s.node = n;
		n.m += flag;
		assignSHierarchy(flag, s, n);
	}
	
	private static void assignSHierarchy(int flag, Sentence s, Node n) {
		n.M += flag;
		for (Word word : s.Words) {
			n.Topics[word.p * s.s].count[word.w] += flag;
			n.Topics[word.p * s.s].n_k += flag;
		}
		if (n.Parent != null) assignSHierarchy(flag, s, n.Parent);		
	}
	
	private static void assignW(int flag, Word w, Sentence s, Node n) {
		Topic topic = n.Topics[s.s * w.p];
		topic.count[w.w] += flag;
		topic.n_k += flag;
		if (n.Parent != null) assignW(flag, w, s, n.Parent);
	}	
}
