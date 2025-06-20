import asyncio
import affine as af


async def main():
    """Main async function."""
    model = "unsloth/gemma-3-4b-it"
    env = af.envs.SAT1(n=5, k=3)
    
    print(f"Generating 10 examples from {env.name}...")
    dataset = await asyncio.gather(*[env.generate() for _ in range(10)])
    
    print(f"Querying {model} on {len(dataset)} examples...")
    responses = await asyncio.gather(*[af.query(d.prompt, model=model) for d in dataset])
    
    print("Validating responses...")
    results = await asyncio.gather(*[d.validate(r) for r, d in zip(responses, dataset)])
    
    print("Saving results...")
    af.save(dataset, responses, results, model=model)
    
    correct_count = sum(1 for r in results if r.get("correct"))
    print(f"Accuracy: {correct_count}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
