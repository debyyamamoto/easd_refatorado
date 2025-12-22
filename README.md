# Algoritmo EASD

## Instruções de Uso

### Parâmetros de Configuração

| Argumento | Flag Curta | Obrigatório | Descrição | Valor Padrão |
|-----------|------------|-------------|-----------|--------------|
| filepath | (N/A) | Sim | Caminho para o arquivo .csv (ex: datasets/cancer.csv) | - |
| time | -time_col | Sim | Nome da coluna de tempo (Survival Time) | - |
| Evento | -event | Sim | Nome da coluna de evento (Status/Censura) | - |
| delimiter | -d | Não | Delimitador do CSV (ex: , ou ;) | , |
| header | --header | Não | Índice da linha de cabeçalho | 0 |
| runs | -r | Não | Número de execuções independentes (loops) | 30 |
| generations | -g | Não | Número máximo de gerações por busca de regra | 500 |
| population | -p | Não | Tamanho da população (indivíduos) | 500 |
| crossover | -c | Não | Taxa de Crossover (0-100) | 50 |
| mutation | -m | Não | Taxa de Mutação (0-100) | 50 |
| restart_check | --restart_check | Não | Percentual de gerações sem melhora para reiniciar | 10 |
| restart_pct | --restart_pct | Não | Percentual da população a ser reiniciada | 10 |
| comparação | --comparacao | Não | Baseline do Log-Rank (complement ou population) | Complement |
| alpha | --a | Não | Peso Alpha para o Fitness | 0.5 |
| executions | --exe | Não | Número de execuções do algoritmo | 1000 |
| ksize | --ksize | Não | Tamanho do rank de Top-K regras | 10 |


### Exemplos de Uso

#### Exemplo 1: Dataset german.txt, coluna alvo 20 e separador de espaço
```bash
python3 main.py datasets/cancer.csv -time time -event status -header 0
```
#### Execução com baseline de população e alpha ajustado
```bash
python3 main.py datasets/cancer.csv -time time -event status -comp population -a 0.8
```

## Estrutura de Saída

Os resultados são salvos em results/[nome_do_dataset]/.

### Arquivos Gerados

**Arquivos por execução:**
- `[dataset][exec#]_DetailedRules.csv`:Todas as regras encontradas, seus intervalos e p-valores.
- `[dataset][exec#]_Info.csv`: Estatísticas detalhadas de cobertura.

**Arquivos agregados:**
- `[dataset]_Mean_Evolution.csv`: Evolução do fitness médio.
- `[dataset]_Best_Evolution.csv`: Evolução do melhor fitness encontrado.
- `[dataset]_FinalResult.txt`: Resumo estatístico médio de todas as 30 (ou N) runs.