import execnet
import sys

def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()


if __name__ == '__main__':
    seed = int(sys.argv[1])
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    result = call_python_version("2.7", "prog", "wrapper", [seed, x, y])
    print('resultado de vuelta')
    print(float(result))
