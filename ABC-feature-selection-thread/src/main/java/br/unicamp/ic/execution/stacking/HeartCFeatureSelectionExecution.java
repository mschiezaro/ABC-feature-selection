package br.unicamp.ic.execution.stacking;

public class HeartCFeatureSelectionExecution extends FeatureSelectionExecution {

	public HeartCFeatureSelectionExecution(boolean[] features) {
		super("heart-c.arff", features, 100, 6, 0.1);
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true};
		FeatureSelectionExecution fs = new HeartCFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
