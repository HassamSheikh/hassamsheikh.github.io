// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "publications",
          description: "Publications in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "PhD in Computer Science (UCF, 2020). SWE-RL at Anyscale. Former Research Scientist at Intel Labs. Publications at ICML, ICLR, IJCNN, AAMAS.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "Research projects and engineering work.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "news-phd-conferred-in-computer-science-from-university-of-central-florida-dissertation-multi-agent-reinforcement-learning-for-defensive-escort-teams",
          title: 'PhD conferred in Computer Science from University of Central Florida. Dissertation: Multi-agent Reinforcement...',
          description: "",
          section: "News",},{id: "news-paper-accepted-at-ijcnn-2022-learning-intrinsic-symbolic-rewards-in-reinforcement-learning",
          title: 'Paper accepted at IJCNN 2022: Learning Intrinsic Symbolic Rewards in Reinforcement Learning.',
          description: "",
          section: "News",},{id: "news-paper-accepted-at-iclr-2022-maximizing-ensemble-diversity-in-deep-reinforcement-learning",
          title: 'Paper accepted at ICLR 2022: Maximizing Ensemble Diversity in Deep Reinforcement Learning.',
          description: "",
          section: "News",},{id: "news-paper-accepted-at-icml-2022-dns-determinantal-point-process-based-neural-network-sampler-for-ensemble-rl",
          title: 'Paper accepted at ICML 2022: DNS — Determinantal Point Process Based Neural Network...',
          description: "",
          section: "News",},{id: "news-joined-anyscale-as-software-engineer-rl-now-the-technical-owner-of-rllib-the-industry-standard-distributed-rl-library",
          title: 'Joined Anyscale as Software Engineer – RL. Now the technical owner of RLlib,...',
          description: "",
          section: "News",},{id: "projects-rllib",
          title: 'RLlib',
          description: "Industry-standard distributed reinforcement learning library",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_rllib/";
            },},{id: "projects-dns",
          title: 'DNS',
          description: "Determinantal Point Process Based Neural Network Sampler for Ensemble RL",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_dns/";
            },},{id: "projects-med-rl",
          title: 'MED-RL',
          description: "Maximizing Ensemble Diversity in Deep Reinforcement Learning",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_medrl/";
            },},{id: "projects-de-maddpg",
          title: 'DE-MADDPG',
          description: "Multi-critic MARL for combined individual and team reward",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_demaddpg/";
            },},{id: "projects-blue-agents",
          title: 'Blue-Agents',
          description: "Modular RL research library for standardizing experimentation at Intel Labs",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_blue_agents/";
            },},{id: "projects-ecnet",
          title: 'ECNet',
          description: "Efficient communication in multi-agent RL via learned communication gates",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_ecnet/";
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
