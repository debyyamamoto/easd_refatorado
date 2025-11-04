# Algoritmo EASD

## Instruções de Uso

### Parâmetros de Configuração

| Argumento | Flag Curta | Obrigatório | Descrição | Valor Padrão |
|-----------|------------|-------------|-----------|--------------|
| filepath | (N/A) | Sim | Caminho para o arquivo do dataset (ex: datasets/Mixed/german.txt) | - |
| target_col | -t | Sim | Índice (base 0) da coluna alvo (Y) | - |
| delimiter | -d | Não | Delimitador do arquivo. Use -d " " para espaços | , |
| header | --header | Não | Índice da linha de cabeçalho. Use --header 0 para primeira linha | None |
| runs | -r | Não | Número de execuções independentes (loops) | 30 |
| generations | -g | Não | Número máximo de gerações por busca de regra | 500 |
| population | -p | Não | Tamanho da população | 500 |
| support | -s | Não | Suporte mínimo da classe (percentual 0.0-1.0) | 0.50 |
| crossover | -c | Não | Taxa de Crossover (0-100) | 50 |
| mutation | -m | Não | Taxa de Mutação (0-100) | 50 |
| restart_check | --restart_check | Não | Percentual de gerações sem melhora para reiniciar | 10 |
| restart_pct | --restart_pct | Não | Percentual da população a ser reiniciada | 10 |

### Exemplos de Uso

#### Exemplo 1: Dataset german.txt, coluna alvo 20 e separador de espaço
```bash
python main.py datasets/Mixed/german.txt -t 20 -d " "
```

## Estrutura de Saída

Todos os resultados são salvos automaticamente na pasta `results/`. Uma subpasta é criada para cada dataset (ex: `results/german/`).

### Arquivos Gerados

**Arquivos por execução:**
- `[dataset][exec#]_DetailedRules.csv`: CSV com detalhes de cada regra encontrada na execução
- `[dataset][exec#]_Info.csv`: CSV com informações de análise das regras

**Arquivos agregados:**
- `[dataset]_Mean_Evolution.csv`: Histórico do fitness médio de todas as gerações e execuções
- `[dataset]_Best_Evolution.csv`: Histórico do melhor fitness de todas as gerações e execuções
- `[dataset]_FinalResult.txt`: Arquivo de resumo com médias e desvios padrão agregados de todas as runs