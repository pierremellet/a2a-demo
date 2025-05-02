# Intégration de MCP et A2A pour une IA Générative Modulaire

## L'essor de l'IA générative

Créer une application avec de l'IA générative implique de fréquentes interactions avec des modèles de langage ou d'autres algorithmes. Récemment, de nombreux frameworks ont émergé pour simplifier l'utilisation de ces modèles.

Face à l'accroissement de la complexité des usages et à l'évolution technologique, il est crucial d'éviter les applications monolithiques. Cela rappelle l'adoption des architectures orientées services, permettant de scinder les applications en unités fonctionnelles réutilisables.

## Du framework de développement à l'écosystème intégré

Les frameworks pour l'IA générative offrent des solutions modulaires et intégrées, allant de la création à la mise en service. Cela inclut des modules pour les prompts, les agents, les fonctions et les sources de données.

Toutefois, avec l'évolution rapide des technologies d'IA Gen, l'interopérabilité entre les solutions est essentielle pour bénéficier de nouvelles opportunités provenant de différents fournisseurs.

## 2025 : vers une standardisation émergente

Début 2025 marque l'essor de protocoles de communication facilitant l'assemblage des composantes techniques d'une application d'IA générative, tout en restant **indépendants des technologies** utilisées.

Deux protocoles se démarquent : MCP, qui apporte des informations contextuelles à un LLM, et A2A (Agent 2 Agent), qui permet la communication entre agents.

L'association de ces protocoles favorise la création d'architectures évolutives et permet de choisir les meilleures solutions disponibles.


# Place à la pratique !

Rien de tel qu'une mise en pratique concrète pour évaluer la maturité et l'intérêt d'une technologie !

Pour cela, imaginons un cas pratique : un assisant conversationnel pouvant faire du **conseil financier**, des **réponses aux questions sur des produits bancaires** et qui permet de **faire une prise de contact**.

``` mermaid
graph TD
    customer["Client"]
    assistant["Assistant conversationnel<br/>Pilote de la conversation"]
    rdv["Prise de rendez-vous"]
    account["Gestion des comptes"]
    qa["Catalague de produits"]

    customer -->|Echange conversationnellement avec l'assistant| assistant
    assistant -->|Identification des disponibilités et prise de rendez-vous| rdv
    assistant -->|Permet d'avoir accès aux informations sur les comptes| account
    assistant -->|Permet d'avoir le détails de prouits| qa

```

Imaginons que la prise de rendez-vous, la gestion des comptes et le catalogue de produits sont des services existants et que je souhaite les expoiter via un assistant conversationnel.

Une première approche pourrait être de considérer l'accès à ces services depuis l'assistant conversationnel en exploitant la capacité de **function calling* des LLM.

Le **function calling** permet à un modèle de langage (LLM) de déclencher des fonctions spécifiques en réponse à un prompt. Lorsqu'on lui fournit une liste de fonctions disponibles avec leurs noms et paramètres, le LLM peut décider d'appeler l'une ou plusieurs de ces fonctions en fournissant les arguments nécessaires. Le composant qui utilise le LLM exécute ensuite ces fonctions et transmet les résultats au modèle pour continuer le processus de génération.

``` mermaid

graph TD
    customer["Client"]

    subgraph assistant["Assistant conversationnel<br/>Pilote de la conversation"]
        orchestrateur["Orchestrateur de la conversation"]
        function_calling_rdv["Functions Calling<br/>Prise de rendez-vous"]
        function_calling_account["Functions Calling<br/>Gestion des comptes"]
        function_calling_qa["Functions Calling<br/>FAQ produits"]
        orchestrateur --> function_calling_rdv
        orchestrateur --> function_calling_account
        orchestrateur --> function_calling_qa
    end

    rdv["Prise de rendez-vous"]
    account["Gestion des comptes"]
    qa["Catalogue de produits"]

    customer -->|Echange conversationnellement avec l'assistant| assistant
    function_calling_rdv-->|Interaction pour RDV| rdv
    function_calling_account -->|Accès aux informations des comptes| account
    function_calling_qa -->|Détails des produits| qa
```
