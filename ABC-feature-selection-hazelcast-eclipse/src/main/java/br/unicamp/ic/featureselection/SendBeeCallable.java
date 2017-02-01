package br.unicamp.ic.featureselection;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Callable;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

import weka.classifiers.Classifier;
import weka.core.Instances;

import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.swarm.FoodSource;

/**
 * Usado para paralelizar o envio de abelhas a uma fonte de alimento
 * 
 */
public class SendBeeCallable implements Callable<BeeParallelExecutionResult>,
		Serializable {

	private static final long serialVersionUID = 41047559459053282L;

	private FoodSource foodSource;

	private Instances originalInstances;

	private Classifier classifier;
	
	private int featureSize;
	
	private int limit;
	
	private double mr;
	
	/**
	 * Estratégia de execução do classificador
	 */
	private static final int KFOLD = 10;

	public SendBeeCallable(FoodSource foodSource, Instances originalInstances,
			Classifier classifier, int limit, double mr) {
		
		this.foodSource = foodSource;
		this.originalInstances = originalInstances;
		featureSize = originalInstances.numAttributes() - 1;
		this.classifier = classifier;
		this.limit = limit;
		this.mr = mr;
	}

	@Override
	public BeeParallelExecutionResult call() throws Exception {
		return sendBee(foodSource);
	}

	/**
	 * Verifica quais features serão utilizadas, eplora a vizinhança, criando
	 * uma variação dessa mesma fonte de alimento, verifica se a fonte pode ser
	 * abandonada (caso seja abandonada é necessário marcar para ser removida e
	 * criar um scout bee para substituir a fonte abandonada) e verifica se a
	 * fonte explorada é melhor que a atual e passa a considera-la para futuras
	 * explorações sempre armazenando a melhor fonte de alimento
	 * 
	 * @param foodSource
	 *            Fonte de alimento a partir da qual a vizinhança será explorada
	 */
	private BeeParallelExecutionResult sendBee(FoodSource foodSource) {

		Set<FoodSource> markedToRemoved = new HashSet<FoodSource>();
		Set<FoodSource> neighbors = new HashSet<FoodSource>();
		Set<FoodSource> abandoned = new HashSet<FoodSource>();
		Set<FoodSource> visitedFoodSources = new HashSet<FoodSource>();
		double bestFitness = 0;
		FoodSource bestFoodSource = null;

		// recupera quais features são utilizadas por essa fonte de alimento
		boolean features[] = foodSource.getFeatureInclusion();
		Set<HazelcastInstance> instances = Hazelcast.getAllHazelcastInstances();
		HazelcastInstance hz = instances.iterator().next();
		Set<FoodSource> foodSources = hz.getSet("Food-Sources");
		// guarda o número de feaures sendo utilizadas
		int nrFeatures = foodSource.getNrFeatures();
		int times = 0;
		FoodSource modifedFoodSource = null;
		do {
			times++;
			// Caso seja modificada apenas umna feature por vez
			for (int i = 0; i < featureSize; i++) {
				if (Math.random() < mr) {
					if (!features[i]) {
						nrFeatures++;
						features[i] = true;
					}
				}
			}
			modifedFoodSource = new FoodSource(features);
		} while ((foodSources.contains(modifedFoodSource)
				|| neighbors.contains(modifedFoodSource)
				|| abandoned.contains(modifedFoodSource) || visitedFoodSources
					.contains(modifedFoodSource)) && times <= featureSize);

		if (!(foodSources.contains(modifedFoodSource)
				|| neighbors.contains(modifedFoodSource)
				|| abandoned.contains(modifedFoodSource) || visitedFoodSources
					.contains(modifedFoodSource))) {

			double fitness = calculateFitness(features);
			modifedFoodSource.setFitness(fitness);
			modifedFoodSource.setNrFeatures(nrFeatures);

			// atual é melhor que a vizinha que está sendo explorada
			if (foodSource.getFitness() > fitness
					|| (fitness == foodSource.getFitness() && nrFeatures > foodSource
							.getNrFeatures())) {

				// incrementa o contador
				foodSource.incrementLimit();
				// se foi explorado mais que limit vezes abandona a fonte
				if (foodSource.getLimit() >= limit) {
					abandoned.add(foodSource);
				}
				visitedFoodSources.add(modifedFoodSource);
				// a vizinha é melhor que a atual
			} else {
				// se o fitness dessa fonte é melhor que o melhor já
				// encontrado
				// armazena, se for igual ao melhor armazenado verifica qual
				// dos
				// dois utliza o menor nr de features
				if (fitness > bestFitness
						|| (fitness == bestFitness && nrFeatures < bestFoodSource
								.getNrFeatures())) {
					bestFoodSource = new FoodSource(modifedFoodSource);
					bestFitness = fitness;
				}
				markedToRemoved.add(foodSource);
				neighbors.add(modifedFoodSource);
			}

		}
		return new BeeParallelExecutionResult(markedToRemoved, neighbors,
				abandoned, visitedFoodSources, bestFoodSource, bestFitness);
	}
	
	/**
	 * Calcula a qualidade de uma fonte de alimento através da fonte selecionada
	 * 
	 * @param features
	 *            Features utilizadas para o cálculo da qualidade da fonte de
	 *            alimento
	 * @return valor de 0 a 100 que indica a qualidade da fonte
	 */
	private double calculateFitness(boolean features[]) {

		ClassifierExecutor executor = new KFoldClassifierExecutor(classifier);
		// carrega as features
		executor.setOriginalInstances(originalInstances);
		executor.loadFeatures();
		// chama o classificador e rtorna a acurácia
		return executor.execute(features, KFOLD);
	}

}