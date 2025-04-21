from scripts.data import Data
from scripts.intro import Intro

if __name__ == "__main__":
    while True:
        print("\U0001F6CD️ Welcome to the E-commerce Customer Retention Analyzer!\n")

        proceed = input("Would you like to perform analysis? (y/n): ").lower().strip()
        if proceed == 'n':
            print("Exiting the program. Goodbye! 👋")
            exit()

        data_loader = Data()

        # Decorate raw_data with cleansed_data
        @data_loader.cleansed_data
        def get_clean_data():
            return data_loader.raw_data()

        df = get_clean_data()

        intro_instance = Intro(df)
        intro_instance.run()

        another = input("\nWould you like to analyze another segment? (y/n): ").lower().strip()
        if another != 'y':
            print("Thank you for using the tool. Goodbye! 👋")
            exit()

#from scripts.data import Data
#from scripts.intro import Intro

#if __name__ == "__main__":
#    data_loader = Data()
    
    # Decorate raw_data with cleansed_data
#    @data_loader.cleansed_data
#    def get_clean_data():
#        return data_loader.raw_data()
    
#    df = get_clean_data()

#    intro_instance = Intro(df)
#    intro_instance.run()




    