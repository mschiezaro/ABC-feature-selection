package br.unicamp.ic.execution.stacking;

public class GlassFeatureSelectionExecution extends FeatureSelectionExecution {

	public GlassFeatureSelectionExecution(boolean[] features) {
		super("glass.arff", features, 100, 6, 0.1);
	}

	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true};
		FeatureSelectionExecution fs = new GlassFeatureSelectionExecution(
				features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
