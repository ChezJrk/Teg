from integrable_program import TegIntegral, TegVariable




if __name__ == '__main__':
    integral = demo()
    print('Symbolic expression:', integral)
    print('Value:', integral.eval())