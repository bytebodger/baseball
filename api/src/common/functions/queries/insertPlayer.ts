import type { Handed } from '../../enums/Handed.js';
import { Table } from '../../enums/Table.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   baseball_reference_id: string,
   bats: Handed,
   name: string,
   throws: Handed,
   time_born: number,
}

export const insertPlayer = async (fields: Fields) => {
   return await insertIntoTable(Table.player, fields);
}