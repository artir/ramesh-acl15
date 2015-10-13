package Hierarchy;

import java.util.HashMap;
import java.util.Map;

import Main.Storage;
import Util.MapUtil;

public class Topic {
	public int[] count;
	public int n_k;
	
	public Topic() {
		n_k = 0;
		count = new int[Storage.V];
	}
	
	public String topWords(int n) {
		Map<String, Integer> tempMap = new HashMap<String, Integer>();
		for (int i = 0; i < Storage.V; i++) {
			tempMap.put(Storage.InvDictionary.get(i), count[i]);
		}
		Map<String, Integer> sorted = MapUtil.sortByValue(tempMap);
		int counter = 0;
		String retValue = "";
		for (String key : sorted.keySet()) {
			int value = sorted.get(key);
			if (value == 0) break;
			retValue += key + " ";
			counter++;
			if (counter == n) break;
		}
		return retValue;
	}
	
	public String toString() {
		return topWords(10);
	}
}
