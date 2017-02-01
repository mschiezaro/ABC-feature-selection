package br.unicamp.ic.featureselection;

import java.util.Collections;
import java.util.HashSet;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.CopyOnWriteArraySet;

import br.unicamp.ic.featureselection.swarm.FoodSource;

public class Teste {

	public static void main(String[] args) {
		
		CopyOnWriteArraySet<FoodSource> set = new CopyOnWriteArraySet<FoodSource>();
		
		FoodSource f1 = new FoodSource();
		f1.setFeatureInclusion(new boolean[]{false, false, false, false});
		f1.setFitness(70.0);
		f1.setNrFeatures(4);
		set.add(f1);
		
		FoodSource f2 = new FoodSource();
		f2.setFeatureInclusion(new boolean[]{false, true, false, false});
		f2.setFitness(70.0);
		f2.setNrFeatures(4);
		set.add(f2);
		
		FoodSource f3 = new FoodSource();
		f3.setFeatureInclusion(new boolean[]{false, false, true, false});
		f3.setFitness(80.0);
		f3.setNrFeatures(4);
		set.add(f3);
		
		FoodSource f4 = new FoodSource();
		f4.setFeatureInclusion(new boolean[]{false, false, false, true});
		f4.setFitness(90.0);
		f4.setNrFeatures(4);
		set.add(f4);
		
		FoodSource f5 = new FoodSource();
		f5.setFeatureInclusion(new boolean[]{true, false, false, false});
		f5.setFitness(70.0);
		f5.setNrFeatures(4);

		
		System.out.println(Collections.max(set).getFitness());
		System.out.println(Collections.min(set).getFitness());
		System.out.println(set.contains(f5));
		System.out.println(set.size());
	}
		
}