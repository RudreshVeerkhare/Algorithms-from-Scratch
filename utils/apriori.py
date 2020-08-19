from itertools import combinations
from collections import defaultdict


class Apriori:
    def __init__(self, min_support = 0.2, min_confidence = 0.5):
        
        # static vars 
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        
        # memory vars
        self.item_support_values = defaultdict()
        self.data = None
        self.uniqueItems = None
        self.total_items = None
        self.final_itemset = None
        self.rules = defaultdict(list)
        pass


    def fit(self, data: list):
        """
        this method calculates the rules based on the given data
        """

        # init 
        self.data = self.processInput(data)
        self.uniqueItems = self.getUniqueItems(data)
        self.total_items = len(data)

        itemSet = self.getItemSet()
        self.final_itemset = itemSet
        rules = self.getRules(itemSet)

        return self

    
    def predict(self, data: list):
        """
        predicts the association according the rules
        """
        results = defaultdict(list)
        for group in data:
            group = tuple(sorted(group))
            if group in self.rules:
                results[group] = self.rules[group]
            else:
                results[group] = list()
        
        return results


    def getRules(self, itemSet: defaultdict):
        """
        This method finds the association rules based on the min_confidence
        """
        items = list(itemSet)
        for item in items:
            powerset = self.getPowerset(item)
            for x in powerset:
                y = set(item) - set(x)
                confidence = self.getConfidence(set(x), y, item)

                if confidence >= self.min_confidence:
                    self.rules[x].append((tuple(y), confidence))
        
        return self.rules

    def getConfidence(self, x: set, y: set, item: tuple):
        """
        This method calculates the comfidence value for given x --> y
        association
        """
        x = tuple(sorted(x))
        item = tuple(sorted(item))
        return self.item_support_values[item] / self.item_support_values[tuple(x)]

    @staticmethod
    def getPowerset(items):
        """
        This method returns powerset of given set
        """
        powerset = list()
        for k in range(1, len(items)):
            powerset.extend(list(combinations(items, k)))
        return powerset


    def getItemSet(self):
        """
        this function will get all the possible pairs based on the count
        and will prune after each step based on min_support
        """
    
        itemSets = None
        k = 1
        while True:
            combs = list(combinations(self.uniqueItems, k))
            counts_dict = self.getCount(combs)
            counts_dict = self.prune(counts_dict)
            if not counts_dict:
                return itemSets
            
            itemSets = counts_dict
            self.item_support_values.update(itemSets)
            k += 1
    
    
    def prune(self, count_dict: defaultdict):
        """
        This function returns filtered itemSets based on the min_support
        Support = Count(itemSet) / total_items
        """

        for itemSet in list(count_dict):
            support = count_dict[itemSet] / self.total_items
            if support < self.min_support:
                count_dict.pop(itemSet)
        
        return count_dict
    

    def getCount(self, combs: list):
        """
        This function returns the count of each itemSet occurence in data
        """
        counts_dict = defaultdict(int)
        for itemSet in combs:
            itemSet = tuple(sorted(itemSet))
            for group in self.data:
                if set(itemSet) <= group:
                    counts_dict[itemSet] += 1

        return counts_dict

    def getUniqueItems(self, data: list):
        """
        this function will return unique items in data.
        """
        keys = set()
        for group in data:
            keys = keys.union(group)
        return keys


    def processInput(self, data: list):
        """
        data should be list of groups
        ex. [(l1, l2), (l1, l3, l2), (l2, l3)]
        """
        return [set(group) for group in data]