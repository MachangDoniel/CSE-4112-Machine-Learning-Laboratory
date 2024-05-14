import pandas as pd

def readData():
    data = pd.read_excel('1907121.xlsx')
    return data

def showFeatureAttribute(feature_attribute):
    for feature, attribute in feature_attribute.items():
        print(f"Unique elements in '{feature}' are {attribute}")
    print()

def extractFeatures(data):
    features = []
    feature_attributes = {}
    for feature in data.columns:
        features.append(feature)
        feature_attributes[feature] = data[feature].unique()
    showFeatureAttribute(feature_attributes)
    return features, feature_attributes

def showFeatureValueCount(feature_value_count):
    for feature, attribute in feature_value_count.items():
        print(f"Feature: {feature}")
        for attr, count in attribute.items():
            print(f"    {attr} : {count}")
        print()

def calculateFeatureValueCount(data):
    feature_value_count = {}
    for feature in data.columns:
        feature_value_count[feature] = data[feature].value_counts()
    showFeatureValueCount(feature_value_count)
    return feature_value_count

def getPriorProbability(data, feature_value_count):
    prior_probability = {}
    result_col = data.columns[-1]  # last column is the result column
    results = data[result_col].unique()
    total_count = len(data)
    for result in results:
        prior_probability[result] = feature_value_count[result_col][result] / total_count
    
    print("Prior Probability:")
    for result, probability in prior_probability.items():
        print(f"    P({result}): {probability}")
    print()
    
    return prior_probability

def getConditionalProbability(data, feature_value_count):
    conditional_probability = {}
    first_col = data.columns[0]     # first column for days
    result_col = data.columns[-1]   # last column is the result column
    results = data[result_col].unique()
    
    for feature in data.columns:
        if feature not in (first_col, result_col):
            unique_attributes = data[feature].unique()
            for attribute in unique_attributes:
                for result in results:
                    count_attribute_in_result = data[(data[feature] == attribute) & (data[result_col] == result)].shape[0]
                    count_result = feature_value_count[result_col][result]
                    conditional_probability[(feature, attribute, result)] = count_attribute_in_result / count_result if count_attribute_in_result > 0 else 0
    
    return conditional_probability

def showAsTable(conditional_probability, prior_probability, features, results):
    for feature in features:
        if feature != features[0] and feature != features[-1]:
            rows = []
            unique_attributes = set(attr for (feat, attr, res) in conditional_probability.keys() if feat == feature)
            for attribute in unique_attributes:
                for result in results:
                    if (feature, attribute, result) in conditional_probability:
                        rows.append([attribute, result, conditional_probability[(feature, attribute, result)]])
            
            cond_prob_df = pd.DataFrame(rows, columns=['Attribute', 'Result', 'Conditional Probability'])
            
            print(f"Conditional Probabilities Lookup Table for '{feature}':")
            if not cond_prob_df.empty:
                print(cond_prob_df.pivot(index='Attribute', columns='Result', values='Conditional Probability').fillna(0))
            else:
                print("No data available")
            print()
    
    prior_prob_df = pd.DataFrame(list(prior_probability.items()), columns=['Result', 'Prior Probability'])
    print("Prior Probabilities Lookup Table:")
    print(prior_prob_df.to_string(index=False))
    print()

def getUserInput(features, feature_attributes):
    user_input = {}
    for feature in features:
        if feature != features[0] and feature != features[-1]:
            print(f"Enter a value for {feature}. Available options are: {feature_attributes[feature]}")
            value = input(f"{feature}: ")
            while value not in feature_attributes[feature]:
                print(f"Invalid value! Available options for {feature} are: {feature_attributes[feature]}")
                value = input(f"{feature}: ")
            user_input[feature] = value
    return user_input

def smoothProbability(probability, total_prob, smoothing_factor=1):
    if probability == 0:
        return smoothing_factor / (total_prob + smoothing_factor)
    else:
        return probability

def calculateInputProbability(user_input, prior_probability, conditional_probability, result_col):
    result_probabilities = {result: prior_probability[result] for result in prior_probability}
    
    for feature, value in user_input.items():
        for result in result_probabilities:
            if (feature, value, result) in conditional_probability:
                result_probabilities[result] *= conditional_probability[(feature, value, result)]
            else:
                result_probabilities[result] *= smoothProbability(0, 0, smoothing_factor=1)  # Apply Laplace Smoothing
    
    print("Probability of the given input:")
    total_prob = sum(result_probabilities.values())
    if total_prob > 0:
        for result, prob in result_probabilities.items():
            user_input_str = ", ".join(f"{feature}={value}" for feature, value in user_input.items())
            print(f"P({result} | ({user_input_str})) = {prob / total_prob}")
        max_result = max(result_probabilities, key=result_probabilities.get)
        print(f"According to Naive Bayes, the result is '{max_result}' with a probability of {result_probabilities[max_result] / total_prob}")
    else:
        print("The given input combination is not possible.")


def main():
    data = readData()
    features, feature_attributes = extractFeatures(data)
    feature_value_count = calculateFeatureValueCount(data)
    prior_probability = getPriorProbability(data, feature_value_count)
    conditional_probability = getConditionalProbability(data, feature_value_count)
    showAsTable(conditional_probability, prior_probability, features, data[features[-1]].unique())
    
    user_input = getUserInput(features, feature_attributes)
    calculateInputProbability(user_input, prior_probability, conditional_probability, features[-1])

main()
