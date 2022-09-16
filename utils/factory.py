from models.base_methods.coil import COIL
from models.base_methods.der import DER
from models.base_methods.ewc import EWC
from models.base_methods.finetune import Finetune
from models.base_methods.gem import GEM
from models.base_methods.icarl import iCaRL
from models.base_methods.lwf import LwF
from models.base_methods.replay import Replay
from models.base_methods.bic import BiC
from models.base_methods.podnet import PODNet
from models.base_methods.wa import WA

from models.base_methods.tpcil import TPCIL
from models.vit import VITLwF
from models.base_methods.lucir import LUCIR
from models.oodcil import OODCIL
from models.prompt import PromptCIL
from models.der_prompt import DERPrompt
from models.base_methods.derplus import DERPLUS
from models.coop_cifar import coop_cifar100
from models.coop_core50 import coop_core50
from models.vit_kprompt_core50 import vitkprompt_core50
from models.coop_domainnet import coop_domainnet
from models.kpvit_domainnet import kpvit_domainnet
from models.lwf_domainnet import LwF_domainnet
from models.ewc_domainnet import EWC_domainnet

from models.ganfake.lwf import LwF_ganfake
from models.ganfake.ewc import EWC_deepfake
from models.ganfake.l2p import l2p_ganfake
from models.ganfake.coop import coop_ganfake
from models.ganfake.icarl import iCaRL_ganfake
from models.ganfake.lucir import LUCIR_GAN
from models.ganfake.vit_kprompt import vitkprompt_deepfake
from models.ganfake.vit_kp_joint import vitkp_joint_deepfake
from models.ganfake.vit_kp_fixfc import vitkp_fixfc_deepfake
from models.ganfake.coop_finetune import coop_ganfake_ft
from models.ganfake.coop_fixfc import coop_fixfc_ganfake
from models.ganfake.coop_joint import coop_joint_ganfake


def get_model(model_name, args):
    name = model_name.lower()
    options = {'icarl': iCaRL,
               'bic': BiC,
               'podnet': PODNet,
               'lwf': LwF,
               'ewc': EWC,
               'wa': WA,
               'der': DER,
               'finetune': Finetune,
               'replay': Replay,
               'gem': GEM,
               'coil': COIL,

               'tpcil': TPCIL,
               'lucir': LUCIR,
               'vit': VITLwF,
               'oodcil': OODCIL,
               'prompt': PromptCIL,
               'derprompt': DERPrompt,
               'derplus': DERPLUS,
               'coop_cifar100': coop_cifar100,

               'coop_core50': coop_core50,
               'vitkprompt_core50': vitkprompt_core50,
               'coop_domainnet': coop_domainnet,
               'kpvit_domainnet': kpvit_domainnet,
               'lwf_domainnet': LwF_domainnet,
               'ewc_domainnet': EWC_domainnet,

               'icarl_gan': iCaRL_ganfake,
               'lucir_gan': LUCIR_GAN,
               'coop_gan': coop_ganfake,
               'lwf_gan': LwF_ganfake,
               'l2p_gan': l2p_ganfake,
               'vitkprompt_gan': vitkprompt_deepfake,
               'coop_ganfake_ft': coop_ganfake_ft,
               'coop_fixfc_ganfake': coop_fixfc_ganfake,
               'coop_joint_ganfake': coop_joint_ganfake,
               'vitkp_joint_deepfake': vitkp_joint_deepfake,
               'vitkp_fixfc_deepfake': vitkp_fixfc_deepfake,
               'ewc_deepfake': EWC_deepfake,

               }
    return options[name](args)

