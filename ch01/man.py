class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def say_hello(self):
        return f"Hello, my name is {self.name}"

    def goodbye(self):
        return "Goodbye" + self.name + "!"

m = Man("John")
print(m.say_hello())
print(m.goodbye()) 