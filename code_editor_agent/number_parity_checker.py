def check_parity():
    num = int(input("Enter a number: "))
    if num % 2 == 0:
        print(f"The number {num} is even.")
    else:
        print(f"The number {num} is odd.")

check_parity()