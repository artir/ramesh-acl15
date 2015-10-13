package Util;

import java.util.Random;

public class Probability {
	
	private static Random RAND = new Random(System.currentTimeMillis());

	public static int SampleLogMultinomial(double[] p) {
		double[] newp = new double[p.length];
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < p.length; i++) {
			if (max < p[i]) max = p[i];
		}
		for (int i = 0; i < p.length; i++) {
			newp[i] = p[i] - max;
			newp[i] = Math.exp(newp[i]);
		}
		return SampleMultinomial(newp);
	}
	
	public static int SampleMultinomial(double[] p) {
		double[] norm = Normalize(p);
		double rand = RAND.nextDouble();
		double sum = 0;
		for (int i = 0; i < norm.length; i++) {
			sum += norm[i];
			if (sum >= rand) return i;
		}
		return p.length-1;
	}
	
	public static double[] Normalize(double[] p) {
		double sum = 0.0;
		double[] normalized = new double[p.length];
		for (int i = 0; i < p.length; i++) sum += p[i];
		for (int i = 0; i < p.length; i++) normalized[i] = p[i] / sum;
		return normalized;
	}
}
