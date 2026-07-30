class person:
    def __call__(self,name):
        print(f"Hello, I am a person named {name}.")
    def hello(self,name):
        print(f"Hello, I am a person named {name}.")

person = person()
person("Alice")  # This will call the __call__ method
person.hello("Bob")  # This will call the hello method