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

for(int i = 0; i < 10; i++) {
ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("mooc-aspect-test")
Logger log = LoggerFactory.getLogger(this.class)

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "mooc-seededlda-baseline-coarse")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel m = new PSLModel(this, data)
m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]



def trainDir = 'data'+java.io.File.separator+'unsupervised'+java.io.File.separator
Partition truthPart = new Partition(0)
Partition targetPart = new Partition(1)
Partition dummy2 = new Partition(201)
for (Predicate p : [coarsetopic])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, truthPart)
	println trainDir+"coarsetopic.txt"
	InserterUtils.loadDelimitedData(insert, trainDir+"coarsetopic.txt");
	insert1 = data.getInserter(p, targetPart)
	println trainDir+"seededldacoarse.txt"
	InserterUtils.loadDelimitedDataTruth(insert1, trainDir+"seedldacoarse.txt");
}

def resultsDB = data.getDatabase(targetPart, [coarsetopic] as Set)
def groundTruthDB = data.getDatabase(truthPart, [coarsetopic] as Set)

def comparator = new MulticlassPredictionComparator(resultsDB)
comparator.setBaseline(groundTruthDB)
comparator.setResultFilter(new MaxValueFilter(coarsetopic, 1))
//comparator.setThreshold(Double.MIN_VALUE)

allGroundings = resultsDB.executeQuery(Queries.getQueryForAllAtoms(coarsetopic))
map = new HashMap<GroundTerm, Integer>()
for(int k = 0;k<allGroundings.size();++k){
	GroundTerm [] grounding = allGroundings.get(k)
	String index = grounding[1].toString()
	if(!map.containsKey(grounding[1])){
		map.put(grounding[1], Integer.parseInt(index))}
}
MulticlassPredictionStatistics stats = comparator.compare(coarsetopic, map,1)

File outputstats_fine = new File("output/unsupervised/baselinecoarsesigstats.txt")

outputstats_fine.append("")
println "confusion matrix"+ stats.getConfusionMatrix()
println "f1 for fientopic " + stats.getF1()
outputstats_fine.append(stats.getF1(0)+"\n")
outputstats_fine.append(stats.getPrecision(0)+"\n")
outputstats_fine.append( stats.getRecall(0)+"\n")

outputstats_fine.append(stats.getF1(1)+"\n")
outputstats_fine.append(stats.getPrecision(1)+"\n")
outputstats_fine.append(stats.getRecall(1)+"\n")

outputstats_fine.append(stats.getF1(2)+"\n")
outputstats_fine.append(stats.getPrecision(2)+"\n")
outputstats_fine.append(stats.getRecall(2)+"\n")

outputstats_fine.append(stats.getF1(3)+"\n")
outputstats_fine.append(stats.getPrecision(3)+"\n")
outputstats_fine.append(stats.getRecall(3)+"\n")

outputstats_fine.append(stats.getF1(4)+"\n")
outputstats_fine.append(stats.getPrecision(4)+"\n")
outputstats_fine.append(stats.getRecall(4)+"\n")

outputstats_fine.append(stats.getF1(5)+"\n")
outputstats_fine.append(stats.getPrecision(5)+"\n")
outputstats_fine.append(stats.getRecall(5)+"\n")
outputstats_fine.append("end"+"\n")

}