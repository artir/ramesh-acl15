package Util;

import Hierarchy.Node;

public class Printer {
	public static void PrintToConsole(Node node, int depth) {
		if (node.M < 10) return;
		System.out.println(node.topWords(10, depth, "  "));
		for (Node child : node.Children) {
			PrintToConsole(child, depth + 1);
		}
	}
}
