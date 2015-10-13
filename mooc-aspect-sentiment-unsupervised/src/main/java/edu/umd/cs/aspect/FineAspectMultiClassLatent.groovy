package edu.umd.cs.aspect

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables
import util.DataOutputter;
import util.FoldUtils;
import util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.em.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics.BinaryClass;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.kernel.predicateconstraint.*


class FineAspectMultiClassLatent{
	public static void main(String[] args)
	{
		for (int d = 1;d<11;++d)
		{
				new File(System.getProperty("user.home")+"output/crossvalidation_outputs/fold"+d).mkdir()
		}
		for(int i = 0; i <10; ++i)
		{
			FineAspectMultiClassLatent a = new FineAspectMultiClassLatent()
			a.pslmodel(i);
		}
	}

	void pslmodel(int cvSet)
	{
		int folds = 10 // number of folds
		double seedRatio = 1 // ratio of observed labels
		Random rand = new Random(0) // used to seed observed data
		double trainTestRatio = 0.9 // ratio of train to test splits (random)
		double filterRatio = 1.0 // ratio of documents to keep (throw away the rest)
		//targetSize = 3000 // target size of snowball sampler
		double explore = 0.001 // prob of random node in snowball sampler
		int numCategories = 7  // number of label categories
		ConfigManager cm = ConfigManager.getManager()
		ConfigBundle config = cm.getBundle("mooc-aspect-test")
		Logger log = LoggerFactory.getLogger(this.class)

		/* Uses H2 as a DataStore and stores it in a temp. directory by default */
		def defaultPath = System.getProperty("java.io.tmpdir")
		String dbpath = config.getString("dbpath", defaultPath + File.separator + "mooc-aspect-test")
		DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)
		String writefolder = "output/crossvalidation_outputs/"
		PSLModel m = new PSLModel(this, data)
		Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
		results.put(config, new ArrayList<DiscretePredictionStatistics>())

		//Predicates

		m.add predicate: "relevance" , types: [ArgumentType.UniqueID]
		m.add predicate: "relevancepred" , types: [ArgumentType.UniqueID]
		m.add predicate: "seededLDA_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "seededLDA_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "seededLDA_3" , types: [ArgumentType.UniqueID]

		m.add predicate: "post" , types: [ArgumentType.UniqueID]
		m.add predicate: "postcategory" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "postfinecategory1" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
		m.add predicate: "coarsepredall" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

		m.add predicate: "finepredall" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "child", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

		/*
		 * Adding rules
		 */

		m.add rule : (~relevancepred(P) & post(P)) >> finetopic(P, data.getUniqueID(0)), weight : 5, squared:true
		
		m.add rule :  (finepredall(P, F) & relevancepred(P)) >> finetopic(P, F), weight : 5, squared:true

		m.add rule :  (coarsepredall(P, C) & child(C, F) ) >> finetopic(P, F), weight : 5, squared:true
		
		m.add rule :  (coarsepredall(P, C) & child(C, F)) >> finetopic(P, F), weight : 10, squared:true
			
		m.add rule :  (coarsepredall(P, C) & coarsetopic(P, C)  & CHILD(C, F)) >> finetopic(P, F), weight : 10, squared:true
		m.add rule :  (coarsepredall(P, C) & finetopic(P, F) & CHILD(C, F)) >> coarsetopic(P, C), weight : 10, squared:true
		
		m.add rule :  (coarsepredall(P, C)) >> coarsetopic(P,C), weight : 10, squared:true
		
		
		m.add rule : (coarsepredall(P, data.getUniqueID(1))) >> (finetopic(P,data.getUniqueID(1)) | finetopic(P,data.getUniqueID(2)) |
		finetopic(P,data.getUniqueID(3)) | finetopic(P,data.getUniqueID(4)) |finetopic(P,data.getUniqueID(5)) |
		finetopic(P,data.getUniqueID(6)) | finetopic(P,data.getUniqueID(7))), constraint:true
//
//		m.add rule : (coarsepred2(P)) >> (finetopic(P,data.getUniqueID(8)) | finetopic(P,data.getUniqueID(11)) |
//		finetopic(P,data.getUniqueID(9)) | finetopic(P,data.getUniqueID(10)) |finetopic(P,data.getUniqueID(12)) |
//		finetopic(P,data.getUniqueID(13)) | finetopic(P,data.getUniqueID(14))), constraint:true


		m.addKernel(new DomainRangeConstraintKernel(finetopic,DomainRangeConstraintType.Functional))


		List<Partition> trainPartition = new ArrayList<Partition>(folds)
		List<Partition> trainLabelsPartition = new ArrayList<Partition>(folds)
		List<Partition> testDataPartition = new ArrayList<Partition>(folds)
		List<Partition> testLabelsPartition = new ArrayList<Partition>(folds)

		/*
		 * Initialize partitions for all cross validation sets
		 */
		for(int initset =0 ;initset<10;++initset)
		{
			trainPartition.add(initset, new Partition(initset))
			trainLabelsPartition.add(initset, new Partition(initset + folds))
			testDataPartition.add(initset, new Partition(initset + 2*folds))
			testLabelsPartition.add(initset, new Partition(initset + 3*folds))
		}


		/*
		 * Set the folder to write into
		 */
		Integer folder = (cvSet+10)%10;
		if (folder ==0) folder = 10
		File file1 = new File(writefolder+"results.txt");
		//		File file2 = new File(filename2);
		File modelfile = new File(writefolder + "model.txt")

		Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			weights.put(k, k.getWeight());
		/*
		 * Train data partition, each partition has 9 folders, one kept aside for testing...
		 *
		 * loading the predicates from the data files into the trainPartition
		 */
		String filename
		Integer trainSet
		int countloadchild=0
		for (trainSet = 1 ; trainSet<=5;++trainSet)
		{
			Integer dirToUse = 0;
			dirToUse = (cvSet+trainSet)%10
			if(dirToUse==0) dirToUse = 10;
			String filepath = ""
			filename = 'data'+java.io.File.separator+'fold'+dirToUse+java.io.File.separator;
			for (Predicate p : [seededLDA_1, seededLDA_2, seededLDA_3, relevancepred])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
						filepath,"\t");
			}
			for (Predicate p : [post, postcategory, postfinecategory1])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedData(data.getInserter(p, trainPartition.get(cvSet)),
						filepath,"\t");

			}
			if(countloadchild==0){
				String childfile = 'data'+java.io.File.separator+"child.txt";
				println "\t\t\tREADING child" +" from "+childfile+"...";
				InserterUtils.loadDelimitedData(data.getInserter(child, trainPartition.get(cvSet)),
						childfile,"\t");
				countloadchild = 1
			}

			for (Predicate p : [coarsepredall])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
						filepath,"\t");
			}
			for (Predicate p : [finepredall])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
						filepath,"\t");
			}

			/*
			 * Load in the ground truth labels for train partition
			 */
			String finetopicpath = filename+"finetopic.txt"
			InserterUtils.loadDelimitedData(data.getInserter(finetopic, trainLabelsPartition.get(cvSet)),
					finetopicpath);
		}

		/*
		 * For test data partition - it needs only one fold in each partition.... Start with 10,1,2,3.... so on.
		 */
		Integer testSet = 0;
		int countloadchild1 = 0
		for (testSet=6;testSet<=10;++testSet){

			int testfolder = (cvSet+testSet)%10
			if(testfolder==0) testfolder = 10;

			String filepath = ""
			filename = 'data'+java.io.File.separator+'fold'+testfolder+java.io.File.separator;
			for (Predicate p : [seededLDA_1, seededLDA_2, seededLDA_3, relevancepred])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
			}
			for (Predicate p : [post, postcategory, postfinecategory1])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
				InserterUtils.loadDelimitedData(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
			}
			if(countloadchild1==0)
			{

				String childfile = 'data'+java.io.File.separator+"child.txt";
				println "\t\t\tREADING child" +" from "+childfile+"... into test";
				InserterUtils.loadDelimitedData(data.getInserter(child, testDataPartition.get(cvSet)),childfile,"\t");
				countloadchild1 = 1
			}
			println "countloadchild"+ countloadchild
			for (Predicate p : [coarsepredall])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
			}
			for (Predicate p : [finepredall])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
			}
			/*
			 * Load in the ground truth labels for train partition
			 */
			String finetopicpath = filename+"finetopic.txt"
			InserterUtils.loadDelimitedData(data.getInserter(finetopic, testLabelsPartition.get(cvSet)), finetopicpath);
		}


		Set toClose = [seededLDA_1, seededLDA_2, seededLDA_3, relevancepred,child, post, postfinecategory1, coarsepredall,
			finepredall] as Set;

		Database trainDB = data.getDatabase(trainPartition.get(cvSet), toClose as Set);
		Database trainLabelsDB = data.getDatabase(trainLabelsPartition.get(cvSet), [finetopic] as Set);

		ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory1))
		println "groundings for all "+ allGroundings.size();
		for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = trainDB.getAtom(finetopic, grounding);
			atom1.setValue(0.0);
			atom1.commitToDB();
		}

		/*
		 * POPULATE TRAINING DATABASE
		 * Get all knows pairs,
		 */
		int rv = 0, ob = 0
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postcategory))
		for (Predicate p : [coarsetopic]) {
			for (int i = 0; i < allGroundings.size(); i++) {
				GroundTerm [] grounding = allGroundings.get(i)
				GroundAtom atom = trainDB.getAtom(p, grounding)
				if (atom instanceof RandomVariableAtom) {
					rv++
					trainDB.commit((RandomVariableAtom) atom);
				}
				else
					ob++
			}
		}
		System.out.println("Saw " + rv + " rvs and " + ob + " obs")
		
		Database testDB = data.getDatabase(testDataPartition.get(cvSet), toClose)
		Database testLabelsDB = data.getDatabase(testLabelsPartition.get(cvSet), [finetopic] as Set);

		allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(postcategory))
		for (Predicate p : [coarsetopic]) {
			for (int i = 0; i < allGroundings.size(); i++) {
				GroundTerm [] grounding = allGroundings.get(i)
				GroundAtom atom = testDB.getAtom(p, grounding)
				if (atom instanceof RandomVariableAtom) {
					testDB.commit((RandomVariableAtom) atom);
				}
			}
		}
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))
		//weight learning
		DualEM weightLearning = new DualEM(m, trainDB, trainLabelsDB, config);

		weightLearning.learn();
		weightLearning.close();
		modelfile.append(m)
		modelfile.append("\n")


		
		allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory1))
		println "groundings for all "+ allGroundings.size();
		for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = testDB.getAtom(finetopic, grounding);
			atom1.setValue(0.0);
			atom1.commitToDB();
		}

		MPEInference mpe = new MPEInference(m, testDB, config)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

		for (GroundAtom atom : Queries.getAllAtoms(testDB, finetopic)){
			file1.append( atom.toString() + "\t" + atom.getValue()+"\n");
		}



		allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
		System.out.println("printing no of atoms in test db " +allGroundings.size());
		testLabelsDB.close()
		def groundTruthDB = data.getDatabase(testLabelsPartition.get(cvSet), [finetopic] as Set)
		DataOutputter.outputPredicate("output" + "/groundTruth" + cvSet + ".node" , groundTruthDB, finetopic, "\t", false, "nodeid\tlabel")

		allGroundings = groundTruthDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
		System.out.println("printing no of atoms in groundtruth db " +allGroundings.size());
		int totalTestExamples = allGroundings.size()
		DataOutputter.outputClassificationPredictions("output" + "/results" + config.getString("name", "") + cvSet + ".txt",
				testDB, finetopic, "\t")

		allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
		for(int k = 0;k<allGroundings.size();++k){
			GroundTerm [] grounding = allGroundings.get(k)
			System.out.println(grounding);
		}
		HashMap <GroundTerm,Integer> map = new HashMap<GroundTerm, Integer>()
		for(int k = 0;k<allGroundings.size();++k){
			GroundTerm [] grounding = allGroundings.get(k)
			String index = grounding[1].toString()
			if(!map.containsKey(grounding[1])){
				map.put(grounding[1], Integer.parseInt(index))}
		}
		def comparator = new MulticlassPredictionComparator(testDB)
		comparator.setBaseline(groundTruthDB)
		comparator.setResultFilter(new MaxValueFilter(finetopic, 1))
		//comparator.setThreshold(Double.MIN_VALUE)
		MulticlassPredictionStatistics stats = comparator.compare(finetopic, map,1)
		File outputstats = new File("output/crossvalidation_outputs/stats"+cvSet+".txt")
		println stats.getAccuracy()

		println "confusion matrix"+ stats.getConfusionMatrix()
		println "f1 for fientopic " + stats.getF1()
		outputstats.append(  "confusion matrix"+ stats.getConfusionMatrix()+"\n")
		outputstats.append( "f1 for fientopic 0 " + stats.getF1(0)+"\n")
		outputstats.append( "precision for fientopic 0 " + stats.getPrecision(0)+"\n")
		outputstats.append( "recall for fientopic 0 " + stats.getRecall(0)+"\n")

		outputstats.append( "f1 for fientopic 1 " + stats.getF1(1)+"\n")
		outputstats.append( "precision for fientopic 1 " + stats.getPrecision(1)+"\n")
		outputstats.append( "recall for fientopic 1 " + stats.getRecall(1)+"\n")

		outputstats.append( "f1 for fientopic (2) " + stats.getF1(2)+"\n")
		outputstats.append( "precision for fientopic (2) " + stats.getPrecision(2)+"\n")
		outputstats.append( "recall for fientopic (2) " + stats.getRecall(2)+"\n")

		outputstats.append( "f1 for fientopic (3) " + stats.getF1(3)+"\n")
		outputstats.append( "precision for fientopic (3) " + stats.getPrecision(3)+"\n")
		outputstats.append( "recall for fientopic (3) " + stats.getRecall(3)+"\n")

		outputstats.append( "f1 for finetopic (4) " + stats.getF1(4)+"\n")
		outputstats.append( "precision for finetopic (4) " + stats.getPrecision(4)+"\n")
		outputstats.append( "recall for finetopic (4) " + stats.getRecall(4)+"\n")

		outputstats.append( "f1 for finetopic (5) " + stats.getF1(5)+"\n")
		outputstats.append( "precision for finetopic (5) " + stats.getPrecision(5)+"\n")
		outputstats.append( "recall for finetopic (5) " + stats.getRecall(5)+"\n")

		outputstats.append( "f1 for finetopic (6) " + stats.getF1(6)+"\n")
		outputstats.append( "precision for finetopic (6) " + stats.getPrecision(6)+"\n")
		outputstats.append( "recall for finetopic (6) " + stats.getRecall(6)+"\n")

		outputstats.append( "f1 for finetopic (7) " + stats.getF1(7)+"\n")
		outputstats.append( "precision for finetopic (7) " + stats.getPrecision(7)+"\n")
		outputstats.append( "recall for finetopic (7) " + stats.getRecall(7)+"\n")

		outputstats.append( "f1 for finetopic (8) " + stats.getF1(8)+"\n")
		outputstats.append( "precision for finetopic (8) " + stats.getPrecision(8)+"\n")
		outputstats.append( "recall for finetopic (8) " + stats.getRecall(8)+"\n")

		outputstats.append( "f1 for finetopic (9) " + stats.getF1(9)+"\n")
		outputstats.append( "precision for finetopic (9) " + stats.getPrecision(9)+"\n")
		outputstats.append( "recall for finetopic (9) " + stats.getRecall(9)+"\n")

		outputstats.append( "f1 for finetopic (10) " + stats.getF1(10)+"\n")
		outputstats.append( "precision for finetopic (10) " + stats.getPrecision(10)+"\n")
		outputstats.append( "recall for finetopic (10) " + stats.getRecall(10)+"\n")

		outputstats.append( "f1 for finetopic (11) " + stats.getF1(11)+"\n")
		outputstats.append( "precision for finetopic (11) " + stats.getPrecision(11)+"\n")
		outputstats.append( "recall for finetopic (11) " + stats.getRecall(11)+"\n")

		outputstats.append( "f1 for finetopic (12) " + stats.getF1(12)+"\n")
		outputstats.append( "precision for finetopic (12) " + stats.getPrecision(12)+"\n")
		outputstats.append( "recall for finetopic (12) " + stats.getRecall(12)+"\n")

		outputstats.append( "f1 for finetopic (13) " + stats.getF1(13)+"\n")
		outputstats.append( "precision for finetopic (13) " + stats.getPrecision(13)+"\n")
		outputstats.append( "recall for finetopic (13) " + stats.getRecall(13)+"\n")

		outputstats.append( "f1 for finetopic (14) " + stats.getF1(14)+"\n")
		outputstats.append( "precision for finetopic (14) " + stats.getPrecision(14)+"\n")
		outputstats.append( "recall for finetopic (14) " + stats.getRecall(14)+"\n")



		//			def comparator = new DiscretePredictionComparator(testDB)
		//
		//			//groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [coarsetopic] as Set)
		//			comparator.setBaseline(groundTruthDB)
		//			comparator.setResultFilter(new MaxValueFilter(finetopic, 1))
		//			comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero
		//
		//			System.out.println("totalTestExamples " + totalTestExamples)
		//			DiscretePredictionStatistics stats = comparator.compare(finetopic, totalTestExamples)
		//			System.out.println("F1 score positive " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
		//
		//			def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
		//			System.out.println("Method " + config.getString("name", "") + ", fold " + cvSet +", acc " + stats.getAccuracy() +
		//					", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
		//					", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
		//					", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
		//
		//
		//			results.get(config).add(cvSet, stats)
		//

		groundTruthDB.close()
		testDB.close()
		trainDB.close()
			
		 
	}
}