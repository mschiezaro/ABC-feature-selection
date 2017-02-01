package br.unicamp.ic.featureselection;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;

import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.swarm.FoodSource;
import br.unicamp.ic.util.FileUtil;

/**
 * Responsável por fazer a seleção de features
 * 
 * @author Mauricio Schiezaro
 */
public class AsyncFeatureSelection2 {

	private BufferedWriter writer;

	/**
	 * Estratégia de execução do classificador
	 */
	private static final int KFOLD = 10;

	/**
	 * Quantidade máximo possível de features
	 */
	private int featureSize;

	/**
	 * Número máximo de vezes que visitaremos uma fonte de alimento antes de
	 * abandoná-la
	 */
	private int limit;

	/**
	 * Número de execuções do algoritmo ABC
	 */
	private int runtime;

	/**
	 * Fitness da melhor fonte de alimento
	 */
	private double bestFitness;

	/**
	 * Melhor fonte de alimento
	 */
	private FoodSource bestFoodSource;

	/**
	 * Parâmetro que controlorá a perturbação de quantas features serão
	 * modificadas ao explorar a vizinhança
	 */
	private double mr;

	/**
	 * Conjunto de fontes de alimento
	 */
	private Set<FoodSource> foodSources;

	/**
	 * Fontes de alimento que já foram visitadas
	 */
	private Set<FoodSource> visitedFoodSources;

	/**
	 * Armazena os Scout Bees e para posteriormente incluí-los como fonte de
	 * alimento
	 */
	private Set<FoodSource> scouts;

	/**
	 * Armazena as fontes abandonadas para posteriormente removê-las como fonte
	 * de alimento
	 */
	private Set<FoodSource> abandoned;

	/**
	 * Indica se na exploração modificaremos apenas uma feature por vez
	 * {@link PerturbationStrategy.CHANGE_ONE_FEATURE} ou através do parâmetro
	 * de perturbação MR {@link PerturbationStrategy.USE_MR}
	 */
	private PerturbationStrategy perturbation;

	/**
	 * Guarda quantas fontes de alimento o algoritmo consultou durante a busca
	 */
	private long states;

	private FileUtil fileUtil;

	private Classifier knnClassifier = new IBk();

	private Instances originalInstances;

	private int threadSize;

	/**
	 * Construtor para seleção de features com a fontes as fontes de alimento
	 * sendo inicializadas uma por feature e utilizando o parâmetro de
	 * perturbação MR que define quantos parâmetros são moficados ao explorar a
	 * vizinhança
	 * 
	 * @param runtime
	 *            Número de execuções do ABC
	 * @param limit
	 *            Máximo número de vezes que será explorado uma vizinhança,
	 *            antes de abandonar a fonte de alimento
	 * @param mr
	 *            Parâmetro de perturbação para definir quantas features serão
	 *            incluídas ou excluídas durante a exploração. Se mr > 0
	 *            perturbation = PerturbationStrategy.USE_MR, se mr <=0
	 *            perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE;
	 * @param executor
	 *            Responsável pela classificação
	 * 
	 */
	public AsyncFeatureSelection2(int runtime, int limit, double mr,
			String databaseName, int threadSize, Filter... filter) {
		this.runtime = runtime;
		this.limit = limit;
		this.mr = mr;
		if (mr > 0) {
			perturbation = PerturbationStrategy.USE_MR;
		} else {
			perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE;
		}
		try {
			// carrega os dados do arquivo
			originalInstances = new Instances(new FileReader(
					System.getProperty("user.dir") + "/src/main/resources/"
							+ databaseName));
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		if (filter != null) {
			for (int i = 0; i < filter.length; i++) {

				try {
					filter[i].setInputFormat(originalInstances);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
				try {
					originalInstances = Filter.useFilter(originalInstances,
							filter[i]);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}

			}
		}
		this.featureSize = originalInstances.numAttributes() - 1;
		this.threadSize = threadSize;
	}

	/**
	 * Construtor para seleção de features com a fontes as fontes de alimento
	 * sendo inicializadas uma por feature e alterando apenas uma feature por
	 * vez ao explorar a vizinhança
	 * 
	 * @param runtime
	 *            Número de execuções do ABC
	 * @param limit
	 *            Máximo número de vezes que será explorado uma vizinhança,
	 *            antes de abandonar a fonte de alimento
	 * @param executor
	 *            Responsável pela classificação
	 */
	public AsyncFeatureSelection2(int runtime, int limit, String databaseName,
			int threadSize, Filter... filter) {

		this(runtime, limit, 0, databaseName, threadSize, filter);
	}

	/**
	 * Executa todo o processo de seleção
	 */
	public void execute() {
		visitedFoodSources = new CopyOnWriteArraySet<FoodSource>();
		abandoned = new CopyOnWriteArraySet<FoodSource>();

		fileUtil = FileUtil.newInstance();
		writer = fileUtil.getWriter();

		bestFitness = 0;

		states = 0;

		logFeatureSeletionInit(runtime, limit, mr, perturbation, featureSize);
		long time = System.currentTimeMillis();

		initializeFoodSources();

		for (int i = 0; i < runtime; i++) {
			scouts = new CopyOnWriteArraySet<FoodSource>();
			sendEmployedBees();
			fileUtil.flush();
			sendOnlookerBees();
			fileUtil.flush();
			sendScoutBeesAndRemoveAbandonsFoodSource();
			fileUtil.flush();
			logIntermediarySolution();
			fileUtil.flush();
		}
		time = (System.currentTimeMillis() - time) / 60000;
		logBestSolutionAndExecutionTime(time);
		states = 0;
	}

	/**
	 * Inicializa as fontes de alimento com o mesmo número de features, cada
	 * fonte tem selecionado apenas uma feature, a medida que o algoritmo é
	 * executado novas features são adicionadas ao conjunto
	 */
	private void initializeFoodSources() {

		System.out.println("initializeFoodSources");

		foodSources = new CopyOnWriteArraySet<FoodSource>();

		ExecutorService executor = Executors.newFixedThreadPool(threadSize);
		CompletionService<FoodSource> completionService = new ExecutorCompletionService<FoodSource>(
				executor);

		int index = 0;

		int loopSize = featureSize / threadSize;

		int rest = featureSize % threadSize;

		for (int i = 0; i < loopSize; i++) {
			for (int j = 0; j < threadSize; j++) {
				states++;
				boolean features[] = new boolean[featureSize];
				features[index++] = true;
				// calcula a qualidade do conjunto criado paralelizando em
				// featureSize threads

				System.out.print(index + " ");
				ClassifierExecutionCallable callable = new ClassifierExecutionCallable(
						features);
				completionService.submit(callable);
			}
			for (int j = 0; j < threadSize; j++) {
				// recupera os valores conforme são retornados da thread
				FoodSource foodSource = null;
				try {
					Future<FoodSource> foodsFuture = completionService.take();
					foodSource = foodsFuture.get();
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				} catch (ExecutionException e) {
					throw new RuntimeException(e);
				}
				// calcula a qualidade do conjunto criado
				double curFitness = foodSource.getFitness();
				// adicionamos fonte de alimento
				foodSources.add(foodSource);
				// armazena a melhor
				if (curFitness > bestFitness) {
					bestFoodSource = new FoodSource(foodSource);
					bestFitness = curFitness;
				}
			}
		}
		
		for (int i = 0; i < rest; i++) {
			states++;
			boolean features[] = new boolean[featureSize];
			features[index++] = true;
			// calcula a qualidade do conjunto criado paralelizando em
			// featureSize threads

			System.out.print(index + " ");
			ClassifierExecutionCallable callable = new ClassifierExecutionCallable(
					features);

			completionService.submit(callable);
			
		}
		for (int i = 0; i < rest; i++) {
			// recupera os valores conforme são retornados da thread
			FoodSource foodSource = null;
			try {
				Future<FoodSource> foodsFuture = completionService.take();
				foodSource = foodsFuture.get();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			} catch (ExecutionException e) {
				throw new RuntimeException(e);
			}
			// calcula a qualidade do conjunto criado
			double curFitness = foodSource.getFitness();
			// adicionamos fonte de alimento
			foodSources.add(foodSource);
			// armazena a melhor
			if (curFitness > bestFitness) {
				bestFoodSource = new FoodSource(foodSource);
				bestFitness = curFitness;
			}
		}
		System.out.println("");
		executor.shutdown();
	}

	/**
	 * Envia as abelhas para explorar as fontes de alimento e sua vizinhança
	 */
	private void sendEmployedBees() {

		System.out.println("sendEmployedBees");

		int foodSourceListSize = foodSources.size();
		System.out.println(foodSourceListSize);

		int index = 0;

		int loopSize = foodSourceListSize / threadSize;

		int rest = foodSourceListSize % threadSize;

		if (foodSourceListSize != 0) {

			ExecutorService executor = Executors.newFixedThreadPool(threadSize);
			CompletionService<BeeParallelExecutionResult> completionService = new ExecutorCompletionService<BeeParallelExecutionResult>(
					executor);

			FoodSource[] sources = foodSources.toArray(new FoodSource[0]);

			for (int i = 0; i < loopSize; i++) {
				for (int j = 0; j < threadSize; j++) {
					SendBeeCallable callable = new SendBeeCallable(
							sources[index++]);
					completionService.submit(callable);
					System.out.print(index + " ");
				}
				for (int j = 0; j < threadSize; j++) {
					try {
						BeeParallelExecutionResult result = completionService
								.take().get();
						foodSources.removeAll(result.getMarkedToRemoved());
						foodSources.addAll(result.getNeighbors());
						abandoned.addAll(result.getAbandoned());
						int abandonedSize = result.getAbandoned().size();
						for (int l = 0; l < abandonedSize; l++) {
							createScoutBee();
						}
						visitedFoodSources.addAll(result
								.getVisitedFoodSources());
						if (result.getBestFitness() > bestFitness
								|| (result.getBestFitness() == bestFitness && result
										.getBestFoodSource().getNrFeatures() < bestFoodSource
										.getNrFeatures())) {

							bestFoodSource = new FoodSource(
									result.getBestFoodSource());
							bestFitness = result.getBestFitness();
						}
					} catch (InterruptedException e) {
						throw new RuntimeException(e);
					} catch (ExecutionException e) {
						throw new RuntimeException(e);
					}

				}
			}
			for (int i = 0; i < rest; i++) {
				SendBeeCallable callable = new SendBeeCallable(sources[index++]);
				completionService.submit(callable);
				System.out.print(index + " ");
			}
			for (int i = 0; i < rest; i++) {
				try {
					BeeParallelExecutionResult result = completionService
							.take().get();
					foodSources.removeAll(result.getMarkedToRemoved());
					foodSources.addAll(result.getNeighbors());
					abandoned.addAll(result.getAbandoned());
					int abandonedSize = result.getAbandoned().size();
					for (int l = 0; l < abandonedSize; l++) {
						createScoutBee();
					}
					visitedFoodSources.addAll(result.getVisitedFoodSources());
					if (result.getBestFitness() > bestFitness
							|| (result.getBestFitness() == bestFitness && result
									.getBestFoodSource().getNrFeatures() < bestFoodSource
									.getNrFeatures())) {

						bestFoodSource = new FoodSource(
								result.getBestFoodSource());
						bestFitness = result.getBestFitness();
					}
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				} catch (ExecutionException e) {
					throw new RuntimeException(e);
				}

			}
			executor.shutdown();
		}
		System.out.println("");
	}

	private void sendOnlookerBees() {

		System.out.println("sendOnlookerBees");

		int foodSourceListSize = foodSources.size();
		
		int index = 0;

		int loopSize = foodSourceListSize / threadSize;

		int rest = foodSourceListSize % threadSize;

		if (foodSourceListSize != 0) {

			Double min = Collections.min(foodSources).getFitness();
			Double range = Collections.max(foodSources).getFitness() - min;
			Random random = new Random();

			ExecutorService executor = Executors.newFixedThreadPool(threadSize);
			CompletionService<BeeParallelExecutionResult> completionService = new ExecutorCompletionService<BeeParallelExecutionResult>(
					executor);

			FoodSource[] sources = foodSources.toArray(new FoodSource[0]);
			
			for (int i = 0; i < loopSize; i++) {

				int onLookersSentSize = 0;
				for (int j = 0; j < threadSize; j++) {
					Double prob = (sources[index].getFitness() - min) / range;
					if (random.nextDouble() < prob) {
						SendBeeCallable callable = new SendBeeCallable(sources[index++]);
						completionService.submit(callable);
						onLookersSentSize++;
					} else {
						sources[index++].incrementLimit();
					}
					System.out.print(index + " ");
				}
				for (int j = 0; j < onLookersSentSize; j++) {
					try {
						BeeParallelExecutionResult result = completionService
								.take().get();
						foodSources.removeAll(result.getMarkedToRemoved());
						foodSources.addAll(result.getNeighbors());
						abandoned.addAll(result.getAbandoned());
						int abandonedSize = result.getAbandoned().size();
						for (int l = 0; l < abandonedSize; l++) {
							createScoutBee();
						}
						visitedFoodSources.addAll(result.getVisitedFoodSources());
						if (result.getBestFitness() > bestFitness
								|| (result.getBestFitness() == bestFitness && result
										.getBestFoodSource().getNrFeatures() < bestFoodSource
										.getNrFeatures())) {

							bestFoodSource = new FoodSource(
									result.getBestFoodSource());
							bestFitness = result.getBestFitness();
						}
					} catch (InterruptedException e) {
						throw new RuntimeException(e);
					} catch (ExecutionException e) {
						throw new RuntimeException(e);
					}
				}
			}
			int onLookersSentSize = 0;
			for (int i = 0; i < rest; i++) {
				Double prob = (sources[index].getFitness() - min) / range;
				if (random.nextDouble() < prob) {
					SendBeeCallable callable = new SendBeeCallable(sources[index++]);
					completionService.submit(callable);
					onLookersSentSize++;
				} else {
					sources[index++].incrementLimit();
				}
				System.out.print(index + " ");
			}
			for (int i = 0; i < onLookersSentSize; i++) {
				try {
					BeeParallelExecutionResult result = completionService
							.take().get();
					foodSources.removeAll(result.getMarkedToRemoved());
					foodSources.addAll(result.getNeighbors());
					abandoned.addAll(result.getAbandoned());
					int abandonedSize = result.getAbandoned().size();
					for (int l = 0; l < abandonedSize; l++) {
						createScoutBee();
					}
					visitedFoodSources.addAll(result.getVisitedFoodSources());
					if (result.getBestFitness() > bestFitness
							|| (result.getBestFitness() == bestFitness && result
									.getBestFoodSource().getNrFeatures() < bestFoodSource
									.getNrFeatures())) {

						bestFoodSource = new FoodSource(
								result.getBestFoodSource());
						bestFitness = result.getBestFitness();
					}
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				} catch (ExecutionException e) {
					throw new RuntimeException(e);
				}
			}
			executor.shutdown();
		}
		System.out.println("");
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

		// guarda o número de feaures sendo utilizadas
		int nrFeatures = foodSource.getNrFeatures();
		int times = 0;
		FoodSource modifedFoodSource = null;
		do {
			times++;
			// Caso seja modificada apenas umna feature por vez
			if (perturbation.equals(PerturbationStrategy.CHANGE_ONE_FEATURE)) {
				int index = (int) Math.round(Math.random() * (featureSize - 1));
				if (!features[index]) {
					nrFeatures++;
					features[index] = true;
				}
				// Features serão removidas ou incluídas controladas pelor
				// parâmetro
				// MR
			} else if (perturbation.equals(PerturbationStrategy.USE_MR)) {
				for (int i = 0; i < featureSize; i++) {
					if (Math.random() < mr) {
						if (!features[i]) {
							nrFeatures++;
							features[i] = true;
						}
					}
				}
			} else {
				throw new RuntimeException("Invalid perturbation type "
						+ perturbation);
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

			// calcula o fitness novo conjunto de features gerado
			states++;
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
						|| (fitness == bestFitness && (bestFoodSource == null || nrFeatures < bestFoodSource
								.getNrFeatures()))) {
					bestFoodSource = new FoodSource(modifedFoodSource);
					bestFitness = fitness;
				} 
				//markedToRemoved.add(foodSource);
				neighbors.add(modifedFoodSource);
			}

		}
		return new BeeParallelExecutionResult(markedToRemoved, neighbors,
				abandoned, visitedFoodSources, bestFoodSource, bestFitness);
	}

	/**
	 * Cria uma nova fonte de alimento pelo scout
	 */
	private void createScoutBee() {
		boolean features[] = new boolean[featureSize];
		Random random = new Random();
		FoodSource foodSource = null;
		int nrFeatures = 0;
		// novas features
		for (int j = 0; j < featureSize; j++) {
			boolean inclusion = random.nextBoolean();
			if (inclusion) {
				nrFeatures++;
			}
			features[j] = inclusion;
		}

		double curFitness = calculateFitness(features);
		// cria uma nova fonte
		foodSource = new FoodSource(features, curFitness, nrFeatures);
		// se não existe ainda adiciona na lista de scout para ser usada
		// posteriormente
		if (!(foodSources.contains(foodSource)
				|| abandoned.contains(foodSource) || visitedFoodSources
					.contains(foodSource))) {
			states++;
			scouts.add(foodSource);
		}
	}

	/**
	 * Após enviar as employed e Onlookers bees remove todas as fontes
	 * abandonadas e adiciona as novas fontes encontradas pelos scout bees
	 */
	private void sendScoutBeesAndRemoveAbandonsFoodSource() {
		foodSources.removeAll(abandoned);
		foodSources.addAll(scouts);
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

		ClassifierExecutor executor = new KFoldClassifierExecutor(knnClassifier);
		// carrega as features
		executor.setOriginalInstances(originalInstances);
		executor.loadFeatures();
		// chama o classificador e rtorna a acurácia
		return executor.execute(features, KFOLD);
	}

	public void logFeatureSeletionInit(int runtime, int limit, double mr,
			PerturbationStrategy perturbation, int nrFeatures) {

		try {
			writer.write("Feature Selection START--------------------------------------------------------------------------------------------------------");

			writer.newLine();
			writer.write("Runtime [" + runtime + "], Limit [" + limit
					+ "], MR [" + mr + "], perturbation [" + perturbation);
			writer.newLine();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}

	public void logBestSolutionAndExecutionTime(long time) {
		try {
			writer.write("Best " + bestFoodSource);
			writer.newLine();
			writer.write("Executado em " + time + " percorrendo " + states
					+ " estados");
			writer.newLine();
			writer.write("Feature Selection END----------------------------------------------------------------------------------------------------------");
			writer.newLine();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}
	
	public void logIntermediarySolution() {
		try {
			writer.write("Intermediary solution " + bestFoodSource);
			writer.newLine();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}

	/**
	 * Usado para paralelizar a execução de classificação
	 * 
	 */
	private class ClassifierExecutionCallable implements Callable<FoodSource> {

		private boolean[] features;

		public ClassifierExecutionCallable(boolean[] features,
				Filter... filters) {
			this.features = features;
		}

		@Override
		public FoodSource call() throws Exception {
			FoodSource f = new FoodSource(features, calculateFitness(features), 1);
			return f;
		}

	}

	/**
	 * Usado para paralelizar o envio de abelhas a uma fonte de alimento
	 * 
	 */
	private class SendBeeCallable implements
			Callable<BeeParallelExecutionResult> {

		private FoodSource foodSource;

		public SendBeeCallable(FoodSource foodSource) {
			this.foodSource = foodSource;
		}

		@Override
		public BeeParallelExecutionResult call() throws Exception {
			return sendBee(foodSource);
		}

	}

}
