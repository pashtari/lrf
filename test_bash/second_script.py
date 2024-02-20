# second_script.py
import sys

def main():
    # Retrieve the arguments passed from the Bash script
    # sys.argv[0] is the script name, so we start from index 1
    arguments = sys.argv[1:]
    
    # Print the arguments
    print("Arguments received:")
    for arg in arguments:
        print(type(arg))
        print(arg)

if __name__ == "__main__":
    main()
