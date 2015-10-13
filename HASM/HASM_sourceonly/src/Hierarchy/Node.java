package Hierarchy;

import java.util.ArrayList;
import java.util.List;

import Main.Storage;

public class Node {
	public static final int NEUTRAL = 0;
	public static final int POSITIVE = 1;
	public static final int NEGATIVE = 2;
	public static final String[] LABELS = {"NEU", "POS", "NEG"};
	
	public int ID;
	public List<Node> Children;
	public Node Parent;
	public Topic[] Topics;
	public int m;
	public int M;
	
	public Node() {
		init();
	}
	
	public Node(Node parent) {
		init();
		Parent = parent;
	}
	
	public void init() {
		Parent = null;
		M = m = 0;
		ID = Storage.NodeCounter++;
		Storage.Nodes.put(ID, this);
		Children = new ArrayList<Node>();
		Topics = new Topic[3];
		for (int i = 0; i < 3; i++) {
			Topics[i] = new Topic();
		}
	}
	
	public Node AddChild() {
		Node newChild = new Node(this);
		Children.add(newChild);
		System.out.println("New child " + newChild.ID + " under " + this.ID);
		return newChild;
	}
	
	public void RemoveMe() {
		for (Topic t : Topics) {
			t.count = null;
			t = null;
		}
		System.out.println("Removing node " + ID);
		Topics = null;
		if (Parent != null) {
			Parent.Children.remove(this);
		}
		Storage.Nodes.remove(ID);
	}
	
	public String toString() {
		return "Node " + ID;
	}
	
	public String topWords(int n, int depth, String tab) {
		String retValue = "";
		for (int i = 0; i < 3; i++) {
			for (int j = 0 ; j < depth; j++) { retValue += tab; } 
			retValue += LABELS[i] + " (" + Topics[i].n_k + ") " + Topics[i].topWords(n) + "\n";
		}
		return retValue;
	}
}
