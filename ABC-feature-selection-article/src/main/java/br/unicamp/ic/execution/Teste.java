package br.unicamp.ic.execution;

import java.util.Arrays;
import java.util.HashSet;
import java.util.SortedSet;
import java.util.TreeSet;

import br.unicamp.ic.featureselection.swarm.FoodSource;

public class Teste {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		boolean[] a = {true, false, false, true};
		FoodSource af = new FoodSource(a);
		boolean[] b = af.getFeatureInclusion();
		b[1] = true;
		System.out.println(Arrays.toString(b));
		System.out.println(Arrays.toString(af.getFeatureInclusion()));
		
		
		
	}

}
