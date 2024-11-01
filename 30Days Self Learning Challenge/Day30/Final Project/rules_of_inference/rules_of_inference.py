# class KnowledgeBase:
#     def __init__(self):
#         self.assertions = []  # Initialize a list to store all statements (assertions)

#     def tell(self, statement):
#         """Add a statement (assertion) to the knowledge base."""
#         self.assertions.append(statement)

#     def ask(self, query):
#         """Check if the query can be inferred from the knowledge base."""
#         # Direct match in assertions
#         if query in self.assertions:
#             return True  # Directly found

#         # Check for implications
#         for statement in self.assertions:
#             if "->" in statement:
#                 antecedent, consequent = statement.split(" -> ")
#                 # Modus Ponens: if we know the antecedent, the consequent is inferred
#                 if antecedent in self.assertions:
#                     if query == consequent:
#                         return True  # The query is confirmed through inference

#         return False  # If query is not found

#     def infer_facts(self):
#         """Use rules of inference to derive new facts."""
#         derived_facts = set()  # Use a set to avoid duplicates
#         changes = True  # Flag to indicate if we have derived new facts

#         while changes:
#             changes = False  # Reset change flag
#             new_facts = set()  # Temporary set to store new derived facts

#             # Check implications
#             for statement in self.assertions:
#                 if "->" in statement:
#                     antecedent, consequent = statement.split(" -> ")
#                     # If the antecedent is known, we can derive the consequent
#                     if antecedent in self.assertions and consequent not in self.assertions:
#                         new_facts.add(consequent)

#             # Check for universal implications
#             for statement in self.assertions:
#                 if statement.startswith("∀"):
#                     if "King(John)" in self.assertions and "King(x)" in statement:
#                         # If John is a king, then he is a person
#                         if "Person(John)" not in self.assertions:
#                             new_facts.add("Person(John)")
#                     if "Person(John)" in self.assertions and "Person(y)" in statement:
#                         # If John is a person, then he is mortal
#                         if "Mortal(John)" not in self.assertions:
#                             new_facts.add("Mortal(John)")

#             # Add newly derived facts to the knowledge base
#             for fact in new_facts:
#                 if fact not in self.assertions:
#                     self.tell(fact)  # Add new derived facts to the assertions
#                     changes = True  # We made a change
#                     derived_facts.add(fact)  # Track the new derived facts

#         return derived_facts  # Return newly inferred facts

#     def existing_knowledge(self):
#         """Display the current state of the knowledge base."""
#         return self.assertions  # Return all current assertions


# # Example usage
# if __name__ == "__main__":
#     kb = KnowledgeBase()

#     # Telling the knowledge base assertions
#     kb.tell("King(John)")  # John is a king
#     kb.tell("Person(Richard)")  # Richard is a person
#     kb.tell("∀x (King(x) -> Person(x))")  # All kings are persons
#     kb.tell("∀y (Person(y) -> Mortal(y))")  # All persons are mortal
#     kb.tell("King(John) -> Person(John)")  # If John is a king, then John is a person

#     # Display existing knowledge
#     print("Current knowledge base:", kb.existing_knowledge())

#     # Inferring facts
#     new_facts = kb.infer_facts()
#     print("Newly inferred facts:", new_facts)  # Should include inferred facts based on logic

#     # Checking if derived facts are in the knowledge base
#     print("Is John Mortal?", kb.ask("Mortal(John)"))  # Expected: True
#     print("Is John a person?", kb.ask("Person(John)"))  # Expected: True
#     print("Is Richard Mortal?", kb.ask("Mortal(Richard)"))  # Expected: False


import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeBase:
    def __init__(self):
        self.assertions = []  # Initialize a list to store all statements (assertions)

    def tell(self, statement):
        """Add a statement (assertion) to the knowledge base."""
        self.assertions.append(statement)

    def ask(self, query):
        """Check if the query can be inferred from the knowledge base."""
        # Direct match in assertions
        if query in self.assertions:
            return True  # Directly found

        # Check for implications
        for statement in self.assertions:
            if "->" in statement:
                antecedent, consequent = statement.split(" -> ")
                # Modus Ponens: if we know the antecedent, the consequent is inferred
                if antecedent in self.assertions:
                    if query == consequent:
                        return True  # The query is confirmed through inference

        return False  # If query is not found

    def infer_facts(self):
        """Use rules of inference to derive new facts."""
        derived_facts = set()  # Use a set to avoid duplicates
        changes = True  # Flag to indicate if we have derived new facts

        while changes:
            changes = False  # Reset change flag
            new_facts = set()  # Temporary set to store new derived facts

            # Check implications
            for statement in self.assertions:
                if "->" in statement:
                    antecedent, consequent = statement.split(" -> ")
                    # If the antecedent is known, we can derive the consequent
                    if antecedent in self.assertions and consequent not in self.assertions:
                        new_facts.add(consequent)

            # Check for universal implications
            for statement in self.assertions:
                if statement.startswith("∀"):
                    if "King(John)" in self.assertions and "King(x)" in statement:
                        # If John is a king, then he is a person
                        if "Person(John)" not in self.assertions:
                            new_facts.add("Person(John)")
                    if "Person(John)" in self.assertions and "Person(y)" in statement:
                        # If John is a person, then he is mortal
                        if "Mortal(John)" not in self.assertions:
                            new_facts.add("Mortal(John)")

            # Add newly derived facts to the knowledge base
            for fact in new_facts:
                if fact not in self.assertions:
                    self.tell(fact)  # Add new derived facts to the assertions
                    changes = True  # We made a change
                    derived_facts.add(fact)  # Track the new derived facts

        return derived_facts  # Return newly inferred facts

    def existing_knowledge(self):
        """Display the current state of the knowledge base."""
        return self.assertions  # Return all current assertions

    def visualize(self):
        """Visualize the knowledge base as a directed graph."""
        G = nx.DiGraph()  # Create a directed graph

        # Add nodes and edges for implications
        for statement in self.assertions:
            if "->" in statement:
                antecedent, consequent = statement.split(" -> ")
                G.add_edge(antecedent.strip(), consequent.strip())  # Add directed edge

        # Add nodes for all assertions
        for assertion in self.assertions:
            G.add_node(assertion.strip())

        # Draw the graph
        pos = nx.spring_layout(G)  # Layout for the graph
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=8, font_weight='bold', arrows=True)
        plt.title("Knowledge Base Visualization")
        plt.show()


# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()

    # Telling the knowledge base assertions
    kb.tell("King(John)")  # John is a king
    kb.tell("Person(Richard)")  # Richard is a person
    kb.tell("∀x (King(x) -> Person(x))")  # All kings are persons
    kb.tell("∀y (Person(y) -> Mortal(y))")  # All persons are mortal
    kb.tell("King(John) -> Person(John)")  # If John is a king, then John is a person

    # Display existing knowledge
    print("Current knowledge base:", kb.existing_knowledge())

    # Inferring facts
    new_facts = kb.infer_facts()
    print("Newly inferred facts:", new_facts)  # Should include inferred facts based on logic

    # Visualizing the knowledge base
    kb.visualize()

    # Checking if derived facts are in the knowledge base
    print("Is John Mortal?", kb.ask("Mortal(John)"))  # Expected: True
    print("Is John a person?", kb.ask("Person(John)"))  # Expected: True
    print("Is Richard Mortal?", kb.ask("Mortal(Richard)"))  # Expected: False


class SimpleKnowledgeBase:
    def __init__(self):
        # Store our facts and rules
        self.facts = []
        
    def add_fact(self, fact):
        """Add a new fact or rule to our knowledge base"""
        print(f"Learning new information: {fact}")
        self.facts.append(fact)
        
    def show_all_knowledge(self):
        """Display everything we know"""
        print("\n=== EVERYTHING I KNOW ===")
        print("Direct Facts:")
        # Show simple facts (those without ->)
        for fact in self.facts:
            if "->" not in fact:
                print(f"  - {fact}")
                
        print("\nRules:")
        # Show rules (those with ->)
        for fact in self.facts:
            if "->" in fact:
                print(f"  - {fact}")
    
    def think(self):
        """Try to figure out new facts based on what we know"""
        print("\n=== THINKING ABOUT WHAT I KNOW ===")
        new_facts = []
        
        # Look at each rule (if X then Y)
        for rule in self.facts:
            if "->" in rule:
                condition, result = rule.split(" -> ")
                
                # If we know the condition is true, then we can conclude the result
                if condition in self.facts and result not in self.facts:
                    print(f"I just figured out: {result}")
                    print(f"Because I know: {condition}")
                    print(f"And I know the rule: {rule}")
                    new_facts.append(result)
        
        # Add all our new facts to what we know
        for fact in new_facts:
            self.facts.append(fact)
            
        if not new_facts:
            print("I couldn't figure out anything new!")
            
        return new_facts
    
    def ask(self, question):
        """Answer a question based on what we know"""
        print(f"\n=== SOMEONE ASKED: {question}? ===")
        
        # First, check if we know this directly
        if question in self.facts:
            print(f"Yes! I know this for a fact.")
            return True
            
        # Then try to figure it out using our rules
        print("Let me think about this...")
        new_facts = self.think()
        
        # Check if we figured it out
        if question in new_facts or question in self.facts:
            print(f"Yes! I figured this out.")
            return True
            
        print("No, I don't know this and I can't figure it out.")
        return False

# Let's use our knowledge base!
def run_example():
    print("=== STARTING KNOWLEDGE BASE EXAMPLE ===\n")
    
    # Create our knowledge base
    kb = SimpleKnowledgeBase()
    
    # Add some basic facts
    print("First, let's teach it some things:\n")
    kb.add_fact("John is a king")
    kb.add_fact("Richard is a person")
    
    # Add some rules
    kb.add_fact("John is a king -> John is a person")  # If John is king, then John is person
    kb.add_fact("Richard is a person -> Richard is mortal")  # If Richard is person, then Richard is mortal
    kb.add_fact("John is a person -> John is mortal")  # If John is person, then John is mortal
    
    # Show everything we know
    kb.show_all_knowledge()
    
    # Try to figure out new things
    print("\nLet's think about what we know...")
    kb.think()
    
    # Ask some questions
    print("\nNow let's ask some questions:")
    questions = [
        "John is a king",
        "John is a person",
        "John is mortal",
        "Richard is a person",
        "Richard is mortal",
        "Richard is a king"
    ]
    
    for question in questions:
        result = kb.ask(question)
        print(f"Final answer: {result}\n")

if __name__ == "__main__":
    run_example()
    
# we want infer if john is person ,computer isonly provided withfacts that john is a king and other statements. example of symbolic ai