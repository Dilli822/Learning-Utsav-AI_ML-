# from sympy import symbols, Or, Not

# def resolution(knowledge_base, query):
#     # Negate the query
#     negated_query = Not(query)
    
#     # Add negated query to the knowledge base
#     clauses = knowledge_base + [negated_query]
    
#     while True:
#         new_clauses = []
#         # Generate all pairs of clauses
#         for i in range(len(clauses)):
#             for j in range(i + 1, len(clauses)):
#                 # Apply resolution
#                 resolvents = resolve(clauses[i], clauses[j])
#                 if resolvents is not None:
#                     new_clauses.extend(resolvents)
        
#         # If a new clause is empty, we found a contradiction
#         if any(clause == True for clause in new_clauses):
#             return True  # The query is provable
        
#         # If no new clauses are generated, we're done
#         new_clauses = set(new_clauses)  # Remove duplicates
#         if new_clauses.issubset(set(clauses)):
#             break
        
#         clauses.extend(new_clauses)  # Add new clauses to the existing set
    
#     return False  # The query is not provable

# def resolve(clause1, clause2):
#     """ Resolves two clauses and returns the resolvents. """
#     resolvents = []
#     literals1 = clause1.args if isinstance(clause1, Or) else [clause1]
#     literals2 = clause2.args if isinstance(clause2, Or) else [clause2]
    
#     for literal1 in literals1:
#         for literal2 in literals2:
#             if literal1 == Not(literal2):
#                 # Create a new clause by combining the remaining literals
#                 new_clause = [l for l in literals1 if l != literal1] + \
#                              [l for l in literals2 if l != literal2]
#                 if new_clause:  # Avoid empty clauses
#                     resolvents.append(Or(*new_clause))
    
#     return resolvents

# # Example usage
# # Define some symptoms
# fever, cough, fatigue = symbols('fever cough fatigue')

# # Define the knowledge base in CNF
# knowledge_base = [
#     Or((fever), (cough)),  # If you have a fever, you might not cough
#     Or(Not(cough), Not(fatigue)),  # If you cough, you might not be fatigued
#     # Add more clauses as needed...
# ]

# # Define a query to resolve
# query = Or(fever, cough)

# # Run resolution
# is_provable = resolution(knowledge_base, query)
# print("The query is provable:", is_provable)

import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Set the speech rate to normal speed
engine.setProperty('rate', 150)  # Adjust the value as needed

# Function to speak text
def speak_text(text):
    engine.say(text)  # Add the text to the queue
    engine.runAndWait()  # Block while processing all currently queued commands

# Example usage
text = "Hello, this is a text-to-speech conversion using Python without saving the audio."

speak_text(text)
