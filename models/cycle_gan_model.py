import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements a modified CycleGAN model for one-way image-to-image translation (A → B).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping.')

        return parser

    def __init__(self, opt):
        """Initialize the modified CycleGAN model."""
        BaseModel.__init__(self, opt)

        # Specify the losses to print
        self.loss_names = ['D_A', 'G_A', 'cycle_A']
        
        # Specify the images to save/display
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        self.visual_names = visual_names_A  # only for A to B
        
        # Only use G_A and D_A for one-way translation
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
        else:
            self.model_names = ['G_A']

        # Define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss() if opt.lambda_identity > 0 else None
            
            # Optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing."""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; only A -> B translation."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_A(self.fake_B)   # G_A(G_A(A)), to calculate cycle loss only for A

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generator G_A only."""
        lambda_A = self.opt.lambda_A
        lambda_idt = self.opt.lambda_identity

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        
        # Forward cycle loss || G_A(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        
        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0

        # Combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_idt_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights."""
        # forward
        self.forward()
        
        # Update G_A
        self.set_requires_grad(self.netD_A, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # Update D_A
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.optimizer_D.step()