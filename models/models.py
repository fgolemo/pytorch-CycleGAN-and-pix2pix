
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
